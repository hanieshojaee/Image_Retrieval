from Feature_Extraction import feature_extraction
from Image_Quary_Selection import image_query_selection
from Distance_Measuring import distance_measuring
from Performance_Evaluation import performance_evaluation

dataset_path = r'E:\Projects\Project_Dr.Pirasteh\Ghazal\Pilot_Project\Dataset\corel1K\image.orig'
feature_extracted_path = r'E:\Projects\Project_Dr.Pirasteh\Ghazal\Pilot_Project\features.h5'

model, features = feature_extraction(dataset_path, feature_extracted_path)
query_images_names, query_images, query_images_features = image_query_selection(dataset_path, model)
all_sorted_dists_for_all_quary_images, all_accepted_images_for_all_quary_images = distance_measuring(dataset_path, features, query_images_features)
precision_and_recall = performance_evaluation(query_images_names, all_sorted_dists_for_all_quary_images, total_retrieved_images=10, total_relevant_retrieved_images=100)
