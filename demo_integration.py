"""
Complete Integration Demo: Knowledge Graph + Rule-Based Validation
Demonstrates the full pipeline matching the architecture diagrams
"""

from knowledge_graph import KnowledgeGraphRetrieval
from rule_based_validation import HierarchicalAdversarialValidator
import sys


def complete_report_refinement_pipeline(initial_report: str):
    """
    Complete pipeline for report refinement following the architecture:
    1. Generate initial report (already done by your model)
    2. Knowledge Graph Integration (RadLex)
    3. Rule-Based Validation
    4. Return refined report
    """
    
    print("\n" + "🏥"*50)
    print("MEDICAL REPORT REFINEMENT PIPELINE")
    print("🏥"*50 + "\n")
    
    print("📥 INITIAL REPORT:")
    print("-" * 100)
    print(initial_report)
    print("-" * 100 + "\n")
    
    # STEP 1: Knowledge Graph Integration (RadLex)
    print("\n" + "="*100)
    print("STEP 1: KNOWLEDGE GRAPH INTEGRATION (RadLex)")
    print("="*100)
    
    try:
        kg_retrieval = KnowledgeGraphRetrieval(reports_path='reports.json')
        kg_result = kg_retrieval.retrieve_and_integrate(initial_report, k=3)
        
        print(f"✅ Knowledge Graph Integration Complete")
        print(f"  • Entities Extracted: {len(kg_result.get('entities', []))}")
        print(f"  • Entities: {', '.join(kg_result.get('entities', []))}")
        print(f"  • Subgraph Nodes: {len(kg_result.get('subgraph_nodes', []))}")
        print(f"  • Embedding Dimension: {kg_result.get('embedding_dim', 0)}")
        
    except Exception as e:
        print(f"⚠️  Knowledge Graph Integration skipped: {e}")
        kg_result = None
    
    # STEP 2: Rule-Based Adversarial Validation
    print("\n" + "="*100)
    print("STEP 2: HIERARCHICAL ADVERSARIAL VALIDATION")
    print("="*100)
    
    validator = HierarchicalAdversarialValidator()
    validation_result = validator.validate_report_comprehensive(initial_report)
    
    print(f"✅ Validation Complete")
    print(f"  • High Quality: {validation_result['is_high_quality']}")
    print(f"  • Quality Level: {validation_result['quality_level']}")
    print(f"  • Rule Validation: {validation_result['rule_validation']['is_valid']}")
    print(f"  • Severity: {validation_result['rule_validation']['severity_level']}")
    print(f"  • Adversarial Score: {validation_result['adversarial_score']:.4f}" if validation_result['adversarial_score'] else "")
    
    # Display violations if any
    if validation_result['rule_validation']['violations']:
        print(f"\n⚠️  VIOLATIONS DETECTED:")
        for violation in validation_result['rule_validation']['violations']:
            print(f"  • {violation['rule_name']}: {', '.join(violation['conflicts'])}")
    
    # STEP 3: Generate Refined Report
    print("\n" + "="*100)
    print("STEP 3: REPORT REFINEMENT")
    print("="*100)
    
    refinement_suggestions = []
    
    # Suggest improvements based on validation
    if not validation_result['is_high_quality']:
        refinement_suggestions.append("Consider addressing rule violations")
    
    if validation_result['rule_validation']['severity_score'] >= 7:
        refinement_suggestions.append("Add urgency markers for high severity findings")
    
    if kg_result and kg_result['status'] == 'success':
        refinement_suggestions.append(f"Integrated knowledge from {len(kg_result.get('subgraph_nodes', []))} related medical entities")
    
    # Create refined report (in production, use LLM to actually refine)
    refined_report = initial_report
    
    if refinement_suggestions:
        print("📝 Refinement Suggestions:")
        for i, suggestion in enumerate(refinement_suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    # FINAL OUTPUT
    print("\n" + "="*100)
    print("📊 FINAL REFINED REPORT")
    print("="*100)
    print(refined_report)
    
    # Quality metrics
    print("\n" + "="*100)
    print("📈 QUALITY METRICS")
    print("="*100)
    print(f"  • Knowledge Integration: {'✅ Enabled' if kg_result else '❌ Disabled'}")
    print(f"  • Rule Compliance: {validation_result['rule_validation']['rules_passed']}/{validation_result['rule_validation']['total_rules_checked']} rules")
    print(f"  • Adversarial Quality Score: {validation_result['adversarial_score']:.2%}" if validation_result['adversarial_score'] else "  • Adversarial Quality Score: N/A")
    print(f"  • Overall Quality: {validation_result['quality_level']}")
    print(f"  • Clinical Severity: {validation_result['rule_validation']['severity_level']}")
    
    print("\n" + "🏥"*50)
    print("✅ PIPELINE COMPLETE!")
    print("🏥"*50 + "\n")
    
    return {
        'original_report': initial_report,
        'refined_report': refined_report,
        'knowledge_graph_result': kg_result,
        'validation_result': validation_result,
        'quality_metrics': {
            'is_high_quality': validation_result['is_high_quality'],
            'quality_level': validation_result['quality_level'],
            'severity': validation_result['rule_validation']['severity_level'],
            'rule_compliance_rate': validation_result['rule_validation']['rules_passed'] / validation_result['rule_validation']['total_rules_checked']
        }
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         MEDICAL REPORT REFINEMENT SYSTEM                                     ║
║         Knowledge Graph Integration + Rule-Based Validation                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test cases
    test_cases = [
        {
            'name': 'Normal Report',
            'report': 'Heart size and pulmonary vasculature are normal. Lungs are clear. No pneumothorax, effusion, or pneumonia.'
        },
        {
            'name': 'Report with Violations',
            'report': 'Heart is normal but shows cardiomegaly. Lungs are clear with evidence of pneumonia and consolidation.'
        },
        {
            'name': 'Critical Finding',
            'report': 'Large pneumothorax noted on the right side with mediastinal shift. Immediate intervention required.'
        },
        {
            'name': 'Moderate Severity',
            'report': 'Moderate pleural effusion with blunting of costophrenic angles. Mild cardiomegaly present. Heart size mildly enlarged.'
        }
    ]
    
    # Run pipeline for each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*100}")
        print(f"# TEST CASE {i}: {test_case['name']}")
        print(f"{'#'*100}\n")
        
        result = complete_report_refinement_pipeline(test_case['report'])
        
        # Brief summary
        print(f"\n📊 SUMMARY FOR TEST CASE {i}:")
        print(f"  Quality: {result['quality_metrics']['quality_level']}")
        print(f"  High Quality: {'✅' if result['quality_metrics']['is_high_quality'] else '❌'}")
        print(f"  Severity: {result['quality_metrics']['severity']}")
        print(f"  Rule Compliance: {result['quality_metrics']['rule_compliance_rate']:.1%}")
        
        if i < len(test_cases):
            input("\n\n⏸️  Press Enter to continue to next test case...\n")
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                     ✅ ALL TEST CASES COMPLETED!                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
