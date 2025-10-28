"""
Knowledge Graph Integration for Medical Report Refinement
Uses Graph Neural Networks (GNN) for knowledge retrieval and integration

This module builds a medical knowledge graph from clinical reports and uses
Graph Attention Networks (GAT) to retrieve and integrate relevant medical knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class MedicalKnowledgeGraph:
    """
    Builds and manages a knowledge graph from medical reports
    Nodes: Clinical entities (findings, diagnoses, anatomical structures)
    Edges: Relationships between entities
    """
    
    def __init__(self, reports_path='reports.json'):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=300)
        self.reports_path = reports_path
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        
    def build_graph_from_reports(self):
        """
        Build knowledge graph from clinical reports
        """
        print("🔨 Building Knowledge Graph from reports...")
        
        # Load reports
        with open(self.reports_path, 'r', encoding='utf-8') as f:
            reports_data = json.load(f)
        
        # Define clinical entity types
        anatomical_structures = ['heart', 'lung', 'lungs', 'cardiac', 'pulmonary', 'chest', 
                                'mediastinum', 'pleural', 'diaphragm', 'ribs', 'thorax']
        
        findings = ['normal', 'abnormal', 'clear', 'enlarged', 'effusion', 'pneumonia',
                   'consolidation', 'opacity', 'pneumothorax', 'atelectasis', 'edema',
                   'cardiomegaly', 'infiltrate', 'nodule', 'mass']
        
        relationships = ['shows', 'indicates', 'suggests', 'reveals', 'presents',
                        'demonstrates', 'displays']
        
        entity_idx = 0
        
        # Extract entities and relationships from reports
        for report_id, report in list(reports_data.items())[:1000]:  # Process first 1000 for efficiency
            abstract = report.get('Abstract', {})
            
            # Combine all text
            findings_text = str(abstract.get('FINDINGS', ''))
            impression_text = str(abstract.get('IMPRESSION', ''))
            combined_text = (findings_text + ' ' + impression_text).lower()
            
            # Extract entities
            found_anatomical = [ent for ent in anatomical_structures if ent in combined_text]
            found_findings = [find for find in findings if find in combined_text]
            
            # Add nodes to graph
            for entity in found_anatomical + found_findings:
                if entity not in self.entity_to_idx:
                    self.graph.add_node(entity, entity_type='clinical_entity')
                    self.entity_to_idx[entity] = entity_idx
                    self.idx_to_entity[entity_idx] = entity
                    entity_idx += 1
            
            # Add edges (relationships)
            for anat in found_anatomical:
                for finding in found_findings:
                    # Add edge from anatomical structure to finding
                    if self.graph.has_node(anat) and self.graph.has_node(finding):
                        if self.graph.has_edge(anat, finding):
                            self.graph[anat][finding]['weight'] += 1
                        else:
                            self.graph.add_edge(anat, finding, weight=1, relation='has_finding')
        
        print(f"✅ Knowledge Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_pyg_data(self):
        """
        Convert NetworkX graph to PyTorch Geometric Data format
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph_from_reports()
        
        # Create node features (using TF-IDF of entity names)
        entities = list(self.entity_to_idx.keys())
        entity_features = self.vectorizer.fit_transform(entities).toarray()
        
        # Pad or truncate to fixed size
        target_dim = 128
        if entity_features.shape[1] < target_dim:
            padding = np.zeros((entity_features.shape[0], target_dim - entity_features.shape[1]))
            entity_features = np.concatenate([entity_features, padding], axis=1)
        else:
            entity_features = entity_features[:, :target_dim]
        
        x = torch.tensor(entity_features, dtype=torch.float)
        
        # Create edge index
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.graph.edges(data=True):
            u_idx = self.entity_to_idx[u]
            v_idx = self.entity_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_attr.append(data.get('weight', 1.0))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    
    def retrieve_relevant_subgraph(self, query_entities: List[str], k=5):
        """
        Retrieve k-hop subgraph around query entities
        """
        subgraph_nodes = set()
        
        for entity in query_entities:
            if entity in self.graph:
                # Get k-hop neighbors
                neighbors = nx.single_source_shortest_path_length(self.graph, entity, cutoff=k)
                subgraph_nodes.update(neighbors.keys())
        
        subgraph = self.graph.subgraph(subgraph_nodes)
        return subgraph


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for knowledge integration
    Uses GAT layers to process medical knowledge graph
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=512, num_heads=4, dropout=0.2):
        super(GraphAttentionNetwork, self).__init__()
        
        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * num_heads)
        
    def forward(self, data):
        """
        Forward pass through GAT layers
        """
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = self.layer_norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Third GAT layer
        x = self.gat3(x, edge_index)
        
        return x


class KnowledgeGraphRetrieval:
    """
    Main class for Knowledge Graph-based retrieval and integration (RadLex integration)
    """
    
    def __init__(self, reports_path='reports.json'):
        self.kg = MedicalKnowledgeGraph(reports_path)
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize(self):
        """
        Initialize the knowledge graph and GNN model
        """
        print("🚀 Initializing Knowledge Graph Retrieval System...")
        
        # Build knowledge graph
        self.kg.build_graph_from_reports()
        
        # Initialize GNN model
        pyg_data = self.kg.get_pyg_data()
        input_dim = pyg_data.x.shape[1]
        
        self.gnn_model = GraphAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=512,
            num_heads=4
        ).to(self.device)
        
        self.gnn_model.eval()
        
        print("✅ Knowledge Graph Retrieval System initialized!")
        
    def extract_entities_from_report(self, report_text: str) -> List[str]:
        """
        Extract clinical entities from generated report
        """
        report_lower = report_text.lower()
        
        # Clinical entities dictionary
        entity_keywords = ['heart', 'lung', 'lungs', 'cardiac', 'pulmonary', 'chest',
                          'normal', 'abnormal', 'clear', 'enlarged', 'effusion', 'pneumonia',
                          'consolidation', 'opacity', 'pneumothorax', 'atelectasis']
        
        found_entities = [entity for entity in entity_keywords if entity in report_lower]
        return found_entities
    
    def retrieve_and_integrate(self, report_text: str, k=5) -> Dict:
        """
        Retrieve relevant knowledge and integrate with GNN
        
        Args:
            report_text: Generated medical report
            k: Number of hops for subgraph retrieval
            
        Returns:
            Dictionary with integrated knowledge and embeddings
        """
        if self.gnn_model is None:
            self.initialize()
        
        # Extract entities from report
        query_entities = self.extract_entities_from_report(report_text)
        
        if not query_entities:
            return {
                'status': 'no_entities_found',
                'entities': [],
                'knowledge_embedding': None
            }
        
        # Retrieve relevant subgraph
        subgraph = self.kg.retrieve_relevant_subgraph(query_entities, k=k)
        
        # Get full graph data and compute embeddings
        pyg_data = self.kg.get_pyg_data().to(self.device)
        
        with torch.no_grad():
            node_embeddings = self.gnn_model(pyg_data)
        
        # Get embeddings for query entities
        entity_embeddings = []
        for entity in query_entities:
            if entity in self.kg.entity_to_idx:
                idx = self.kg.entity_to_idx[entity]
                entity_embeddings.append(node_embeddings[idx].cpu().numpy())
        
        # Aggregate embeddings
        if entity_embeddings:
            aggregated_embedding = np.mean(entity_embeddings, axis=0)
        else:
            aggregated_embedding = None
        
        return {
            'status': 'success',
            'entities': query_entities,
            'subgraph_nodes': list(subgraph.nodes()),
            'subgraph_edges': list(subgraph.edges()),
            'knowledge_embedding': aggregated_embedding,
            'embedding_dim': len(aggregated_embedding) if aggregated_embedding is not None else 0
        }
    
    def print_knowledge_integration(self, report_text: str):
        """
        Beautifully print knowledge graph integration results
        """
        print("\n" + "="*100)
        print("🧠 KNOWLEDGE GRAPH INTEGRATION (RadLex)")
        print("="*100)
        
        result = self.retrieve_and_integrate(report_text)
        
        print(f"\n📄 Input Report:\n{report_text}\n")
        print("-"*100)
        
        if result['status'] == 'success':
            print(f"\n✅ Status: {result['status']}")
            print(f"\n🔍 Extracted Entities ({len(result['entities'])}):")
            for entity in result['entities']:
                print(f"  • {entity}")
            
            print(f"\n🕸️  Retrieved Subgraph:")
            print(f"  • Nodes: {len(result['subgraph_nodes'])}")
            print(f"  • Edges: {len(result['subgraph_edges'])}")
            print(f"  • Connected entities: {', '.join(result['subgraph_nodes'][:10])}")
            
            if result['knowledge_embedding'] is not None:
                print(f"\n🎯 Knowledge Embedding:")
                print(f"  • Dimension: {result['embedding_dim']}")
                print(f"  • Mean: {np.mean(result['knowledge_embedding']):.4f}")
                print(f"  • Std: {np.std(result['knowledge_embedding']):.4f}")
        else:
            print(f"\n⚠️  Status: {result['status']}")
            print("  No clinical entities found in the report.")
        
        print("\n" + "="*100)
        print("✅ Knowledge Integration Complete!")
        print("="*100 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize system
    kg_retrieval = KnowledgeGraphRetrieval(reports_path='reports.json')
    kg_retrieval.initialize()
    
    # Test with sample report
    sample_report = "Heart size and pulmonary vasculature are normal. No evidence of pneumonia, pleural effusion, or pneumothorax. Lungs are clear."
    
    # Perform knowledge graph integration
    kg_retrieval.print_knowledge_integration(sample_report)
    
    # Get programmatic result
    result = kg_retrieval.retrieve_and_integrate(sample_report, k=3)
    print("\n📊 Programmatic Result:")
    print(f"Entities: {result['entities']}")
    print(f"Embedding shape: {result['knowledge_embedding'].shape if result['knowledge_embedding'] is not None else 'None'}")
