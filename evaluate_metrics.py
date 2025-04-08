from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


generated_reports = [
    "the heart size and pulmonary vascularity appear within normal limits no focal consolidation pleural effusion or pneumothorax identified no acute bony findings.",
    "Tthe heart is normal in size and contour the lungs are clear without focal airspace consolidation pleural effusion or pneumothorax is seen no acute bony.",
    "No acute cardiopulmonary abnormality is seen."
]

ground_truth_reports = [
    "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax.",
    "Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs.",
    "No acute findings in the lungs or heart."
]


meteor_scores = []

for ref, pred in zip(ground_truth_reports, generated_reports):
    score = meteor_score([ref.lower().split()], pred.lower().split())
    meteor_scores.append(score)


average_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"Average METEOR Score: {average_meteor  :.4f}")
