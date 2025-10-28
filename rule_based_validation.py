"""
Rule-Based Clinical Validation with Adversarial Learning
Implements hierarchical adversarial validation for medical report quality control

This module enforces clinical rules and uses adversarial validation (discriminators)
to ensure reports meet medical standards and severity alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Clinical severity levels"""
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class ClinicalRule:
    """
    Represents a clinical validation rule
    """
    rule_id: str
    rule_name: str
    rule_description: str
    severity: SeverityLevel
    pattern: str
    required_terms: List[str]
    conflicting_terms: List[str]
    severity_indicators: List[str]


class ClinicalRuleBase:
    """
    Rule-based system for clinical report validation
    Defines medical knowledge rules for chest X-ray reports
    """
    
    def __init__(self):
        self.rules = self._define_rules()
        self.severity_matrix = self._build_severity_matrix()
        
    def _define_rules(self) -> Dict[str, ClinicalRule]:
        """
        Define comprehensive clinical validation rules
        """
        rules = {
            'normal_consistency': ClinicalRule(
                rule_id='R001',
                rule_name='Normal Report Consistency',
                rule_description='If heart is normal, no cardiomegaly should be mentioned',
                severity=SeverityLevel.NORMAL,
                pattern=r'(heart.*normal|normal.*heart)',
                required_terms=['normal', 'heart'],
                conflicting_terms=['cardiomegaly', 'enlarged', 'hypertrophy'],
                severity_indicators=[]
            ),
            
            'pneumonia_indicators': ClinicalRule(
                rule_id='R002',
                rule_name='Pneumonia Detection Rule',
                rule_description='Pneumonia requires consolidation or infiltrate',
                severity=SeverityLevel.MODERATE,
                pattern=r'pneumonia',
                required_terms=['pneumonia'],
                conflicting_terms=['clear', 'normal lungs'],
                severity_indicators=['consolidation', 'infiltrate', 'opacity']
            ),
            
            'effusion_consistency': ClinicalRule(
                rule_id='R003',
                rule_name='Pleural Effusion Rule',
                rule_description='Effusion should mention fluid or pleural space',
                severity=SeverityLevel.MODERATE,
                pattern=r'effusion',
                required_terms=['effusion'],
                conflicting_terms=['no effusion', 'clear'],
                severity_indicators=['fluid', 'pleural', 'blunting']
            ),
            
            'pneumothorax_emergency': ClinicalRule(
                rule_id='R004',
                rule_name='Pneumothorax Critical Rule',
                rule_description='Pneumothorax is critical finding',
                severity=SeverityLevel.CRITICAL,
                pattern=r'pneumothorax',
                required_terms=['pneumothorax'],
                conflicting_terms=['no pneumothorax'],
                severity_indicators=['air', 'collapse', 'tension']
            ),
            
            'cardiomegaly_rule': ClinicalRule(
                rule_id='R005',
                rule_name='Cardiomegaly Rule',
                rule_description='Enlarged heart should not coexist with normal heart',
                severity=SeverityLevel.MODERATE,
                pattern=r'cardiomegaly|enlarged.*heart',
                required_terms=['cardiomegaly', 'enlarged'],
                conflicting_terms=['normal heart', 'heart normal'],
                severity_indicators=['cardiothoracic ratio', 'CTR']
            ),
            
            'clear_lungs_rule': ClinicalRule(
                rule_id='R006',
                rule_name='Clear Lungs Consistency',
                rule_description='Clear lungs cannot have active disease',
                severity=SeverityLevel.NORMAL,
                pattern=r'lungs.*clear|clear.*lungs',
                required_terms=['clear', 'lungs'],
                conflicting_terms=['pneumonia', 'consolidation', 'opacity', 'infiltrate'],
                severity_indicators=[]
            ),
            
            'negative_report_rule': ClinicalRule(
                rule_id='R007',
                rule_name='Negative Report Rule',
                rule_description='Negative reports should explicitly state no abnormalities',
                severity=SeverityLevel.NORMAL,
                pattern=r'no (acute|evidence|pneumonia|effusion)',
                required_terms=['no'],
                conflicting_terms=[],
                severity_indicators=[]
            ),
            
            'severity_alignment': ClinicalRule(
                rule_id='R008',
                rule_name='Severity Alignment Rule',
                rule_description='Findings severity must match impression severity',
                severity=SeverityLevel.MODERATE,
                pattern=r'',
                required_terms=[],
                conflicting_terms=[],
                severity_indicators=['severe', 'critical', 'mild', 'moderate']
            )
        }
        
        return rules
    
    def _build_severity_matrix(self) -> Dict:
        """
        Build severity scoring matrix for different findings
        """
        return {
            # Normal findings (score: 0)
            'normal': 0, 'clear': 0, 'unremarkable': 0, 'no acute': 0,
            
            # Mild findings (score: 1-2)
            'minimal': 1, 'small': 1, 'trace': 1, 'slight': 1,
            
            # Moderate findings (score: 3-5)
            'pneumonia': 3, 'effusion': 3, 'consolidation': 4, 'cardiomegaly': 3,
            'infiltrate': 3, 'opacity': 3, 'atelectasis': 3,
            
            # Severe findings (score: 6-8)
            'large effusion': 6, 'extensive': 7, 'massive': 7,
            
            # Critical findings (score: 9-10)
            'pneumothorax': 9, 'tension pneumothorax': 10, 'mediastinal shift': 9,
            'large pneumothorax': 10
        }
    
    def validate_report(self, report_text: str) -> Dict:
        """
        Validate report against all clinical rules
        
        Returns:
            Dictionary with validation results and violations
        """
        report_lower = report_text.lower()
        violations = []
        passed_rules = []
        severity_score = 0
        
        for rule_id, rule in self.rules.items():
            # Check if rule applies
            if rule.pattern and not re.search(rule.pattern, report_lower):
                continue
            
            # Check for required terms
            has_required = all(term in report_lower for term in rule.required_terms)
            
            if has_required:
                # Check for conflicting terms (violations)
                conflicts = [term for term in rule.conflicting_terms if term in report_lower]
                
                if conflicts:
                    violations.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.rule_name,
                        'severity': rule.severity.name,
                        'description': rule.rule_description,
                        'conflicts': conflicts
                    })
                else:
                    passed_rules.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.rule_name,
                        'severity': rule.severity.name
                    })
        
        # Calculate severity score
        for term, score in self.severity_matrix.items():
            if term in report_lower:
                severity_score = max(severity_score, score)
        
        # Determine overall severity level
        if severity_score == 0:
            overall_severity = SeverityLevel.NORMAL
        elif severity_score <= 2:
            overall_severity = SeverityLevel.MILD
        elif severity_score <= 5:
            overall_severity = SeverityLevel.MODERATE
        elif severity_score <= 8:
            overall_severity = SeverityLevel.SEVERE
        else:
            overall_severity = SeverityLevel.CRITICAL
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'passed_rules': passed_rules,
            'severity_score': severity_score,
            'severity_level': overall_severity.name,
            'total_rules_checked': len(self.rules),
            'rules_passed': len(passed_rules),
            'rules_violated': len(violations)
        }


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial discriminator network for report quality validation
    Learns to distinguish between high-quality and low-quality reports
    """
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64]):
        super(AdversarialDiscriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        Returns probability that input is a high-quality report
        """
        return self.network(x)


class HierarchicalAdversarialValidator:
    """
    Complete hierarchical adversarial validation system
    Combines rule-based validation with adversarial learning
    """
    
    def __init__(self, device='cpu'):
        self.rule_base = ClinicalRuleBase()
        self.discriminator = AdversarialDiscriminator(input_dim=512)
        self.device = torch.device(device)
        self.discriminator.to(self.device)
        self.discriminator.eval()
        
        # Initialize with random weights (in production, load pretrained weights)
        self._initialize_discriminator()
    
    def _initialize_discriminator(self):
        """
        Initialize discriminator with pretrained weights
        In production, this would load from checkpoint
        """
        # For now, just use random initialization
        # In real system: self.discriminator.load_state_dict(torch.load('discriminator.pth'))
        pass
    
    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """
        Convert text to embedding (simplified)
        In production, use proper text encoder (BERT, BioClinicalBERT, etc.)
        """
        # Simple hash-based embedding for demonstration
        words = text.lower().split()
        embedding = np.zeros(512)
        
        for i, word in enumerate(words[:512]):
            embedding[i % 512] += hash(word) % 100 / 100.0
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    
    def validate_report_comprehensive(self, report_text: str, use_adversarial=True) -> Dict:
        """
        Comprehensive validation using both rules and adversarial network
        
        Args:
            report_text: Medical report text
            use_adversarial: Whether to use adversarial discriminator
            
        Returns:
            Complete validation results
        """
        # Rule-based validation
        rule_results = self.rule_base.validate_report(report_text)
        
        # Adversarial validation
        adversarial_score = None
        if use_adversarial:
            embedding = self._text_to_embedding(report_text).to(self.device)
            with torch.no_grad():
                adversarial_score = self.discriminator(embedding).item()
        
        # Combine results
        is_high_quality = (
            rule_results['is_valid'] and 
            (adversarial_score is None or adversarial_score > 0.5)
        )
        
        return {
            'is_high_quality': is_high_quality,
            'rule_validation': rule_results,
            'adversarial_score': adversarial_score,
            'quality_level': self._determine_quality_level(rule_results, adversarial_score)
        }
    
    def _determine_quality_level(self, rule_results: Dict, adversarial_score: float) -> str:
        """
        Determine overall quality level
        """
        rule_pass_rate = rule_results['rules_passed'] / max(rule_results['total_rules_checked'], 1)
        
        if adversarial_score is not None:
            combined_score = (rule_pass_rate + adversarial_score) / 2
        else:
            combined_score = rule_pass_rate
        
        if combined_score >= 0.9:
            return "EXCELLENT"
        elif combined_score >= 0.75:
            return "GOOD"
        elif combined_score >= 0.6:
            return "ACCEPTABLE"
        elif combined_score >= 0.4:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"
    
    def print_validation_results(self, report_text: str):
        """
        Beautifully print validation results
        """
        print("\n" + "="*100)
        print("🛡️  HIERARCHICAL ADVERSARIAL VALIDATION")
        print("="*100)
        
        results = self.validate_report_comprehensive(report_text)
        
        print(f"\n📄 Report:\n{report_text}\n")
        print("-"*100)
        
        # Overall quality
        quality_emoji = {
            "EXCELLENT": "🌟",
            "GOOD": "✅",
            "ACCEPTABLE": "👍",
            "NEEDS_IMPROVEMENT": "⚠️",
            "POOR": "❌"
        }
        
        print(f"\n🎯 Overall Quality: {quality_emoji.get(results['quality_level'], '❓')} {results['quality_level']}")
        print(f"✅ High Quality: {results['is_high_quality']}")
        
        # Rule-based validation
        rule_results = results['rule_validation']
        print(f"\n📋 RULE-BASED VALIDATION:")
        print(f"  • Valid: {rule_results['is_valid']}")
        print(f"  • Severity: {rule_results['severity_level']} (Score: {rule_results['severity_score']})")
        print(f"  • Rules Passed: {rule_results['rules_passed']}/{rule_results['total_rules_checked']}")
        print(f"  • Rules Violated: {rule_results['rules_violated']}")
        
        if rule_results['violations']:
            print(f"\n⚠️  VIOLATIONS:")
            for violation in rule_results['violations']:
                print(f"  • [{violation['rule_id']}] {violation['rule_name']}")
                print(f"    - Severity: {violation['severity']}")
                print(f"    - Conflicts: {', '.join(violation['conflicts'])}")
                print(f"    - Description: {violation['description']}")
        
        if rule_results['passed_rules']:
            print(f"\n✅ PASSED RULES:")
            for passed in rule_results['passed_rules'][:5]:  # Show first 5
                print(f"  • [{passed['rule_id']}] {passed['rule_name']} ({passed['severity']})")
        
        # Adversarial validation
        if results['adversarial_score'] is not None:
            score = results['adversarial_score']
            print(f"\n🤖 ADVERSARIAL DISCRIMINATOR:")
            print(f"  • Quality Score: {score:.4f}")
            print(f"  • Classification: {'HIGH QUALITY' if score > 0.5 else 'LOW QUALITY'}")
            
            # Visual bar
            bar_length = 50
            filled = int(bar_length * score)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  • Visual: [{bar}] {score*100:.1f}%")
        
        print("\n" + "="*100)
        print("✅ Validation Complete!")
        print("="*100 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = HierarchicalAdversarialValidator()
    
    # Test with various reports
    test_reports = [
        # Good report
        "Heart size and pulmonary vasculature are normal. Lungs are clear. No pneumothorax, effusion, or pneumonia.",
        
        # Report with violations
        "Heart is normal but shows cardiomegaly. Lungs are clear with evidence of pneumonia.",
        
        # Critical finding
        "Large pneumothorax noted on the right side with mediastinal shift. Immediate intervention required.",
        
        # Moderate severity
        "Moderate pleural effusion with blunting of costophrenic angles. Mild cardiomegaly present."
    ]
    
    print("🧪 Testing Hierarchical Adversarial Validation\n")
    
    for i, report in enumerate(test_reports, 1):
        print(f"\n{'='*100}")
        print(f"TEST CASE #{i}")
        print(f"{'='*100}")
        validator.print_validation_results(report)
