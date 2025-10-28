"""
Test script to demonstrate finding similar reports functionality
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_k_most_relevant_reports(query_text, k=5, reports_path="reports.json"):
    """
    Find k most similar reports from reports.json using TF-IDF and cosine similarity
    """
    try:
        # Load reports
        with open(reports_path, 'r', encoding='utf-8') as f:
            reports_data = json.load(f)
        
        # Extract report IDs and text content
        report_ids = []
        report_texts = []
        
        for report_id, report in reports_data.items():
            report_ids.append(report_id)
            
            # Combine all abstract fields into one text
            abstract = report.get("Abstract", {})
            combined_text = " ".join([
                str(abstract.get("COMPARISON") or ""),
                str(abstract.get("INDICATION") or ""),
                str(abstract.get("FINDINGS") or ""),
                str(abstract.get("IMPRESSION") or "")
            ])
            report_texts.append(combined_text)
        
        # Add query text to the corpus
        all_texts = report_texts + [query_text]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between query and all reports
        query_vector = tfidf_matrix[-1]
        report_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, report_vectors).flatten()
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            report_id = report_ids[idx]
            similarity_score = similarities[idx]
            report_data = reports_data[report_id]
            results.append((report_id, similarity_score, report_data))
        
        return results
        
    except Exception as e:
        print(f"❌ Error finding similar reports: {e}")
        return []


def print_similar_reports(query_text, k=5):
    """
    Find and beautifully print k most similar reports to console
    """
    print("\n" + "="*100)
    print(f"🔍 FINDING {k} MOST RELEVANT REPORTS")
    print("="*100)
    print(f"\n📄 Query Report:\n{query_text}\n")
    print("-"*100)
    
    similar_reports = find_k_most_relevant_reports(query_text, k)
    
    if not similar_reports:
        print("❌ No similar reports found or error occurred.")
        return
    
    print(f"\n✨ TOP {k} MOST SIMILAR REPORTS:\n")
    
    for rank, (report_id, similarity, report_data) in enumerate(similar_reports, 1):
        print(f"\n{'='*100}")
        print(f"🏆 RANK #{rank} | Report ID: {report_id} | Similarity Score: {similarity:.4f}")
        print(f"{'='*100}")
        
        # Print basic info
        print(f"📅 Date: {report_data.get('Date', 'N/A')}")
        print(f"🏥 Specialty: {report_data.get('Specialty', 'N/A')}")
        print(f"👨‍⚕️ Authors: {', '.join(report_data.get('Authors', ['N/A']))}")
        
        # Print abstract sections
        abstract = report_data.get('Abstract', {})
        print(f"\n📋 ABSTRACT:")
        print(f"  • COMPARISON: {abstract.get('COMPARISON', 'N/A')}")
        print(f"  • INDICATION: {abstract.get('INDICATION', 'N/A')}")
        print(f"  • FINDINGS: {abstract.get('FINDINGS', 'N/A')}")
        print(f"  • IMPRESSION: {abstract.get('IMPRESSION', 'N/A')}")
        
        # Print images
        images = report_data.get('Images', [])
        if images:
            print(f"\n🖼️  Images: {', '.join(images)}")
    
    print("\n" + "="*100)
    print("✅ Search Complete!")
    print("="*100 + "\n")


if __name__ == "__main__":
    # Test with a sample query
    sample_query = "Heart size and pulmonary vasculature are normal. No evidence of pneumonia, pleural effusion, or pneumothorax."
    
    print("\n🚀 Testing Similar Reports Finder\n")
    print_similar_reports(sample_query, k=5)
