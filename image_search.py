import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image
from PIL import Image as PILImage
from intro_multimodal_rag_utils import create_get_metadata_df
from intro_multimodal_rag_utils import (
    display_images,
    get_gemini_response,
    get_similar_image_from_query,
    get_similar_text_from_query,
    print_text_to_image_citation,
    print_text_to_text_citation,
)

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "global")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
text_model = GenerativeModel("gemini-2.5-flash")
multimodal_model = text_model
multimodal_model_flash = text_model
path = "data2/"
text_metadata_df, image_metadata_df = create_get_metadata_df(path,model=multimodal_model)


image_query_path = "tac_table_revenue.png"

# Print a message indicating the input image
print("***Input image from user:***")

# Display the input image
Image.load_from_file(image_query_path)


matching_results_image = get_similar_image_from_query(
    text_metadata_df,
    image_metadata_df,
    # query=query,  # Use query text for additional filtering (optional)
    column_name="mm_embedding_from_img_only",  # Use image embedding for similarity calculation
    image_emb=True,
    image_query_path=image_query_path,  # Use input image for similarity calculation
    top_n=3,  # Retrieve top 3 matching images
    embedding_size=1408,  # Use embedding size of 1408
)

print("\n **** Result: ***** \n")

# Display the Top Matching Image
img = PILImage.open(matching_results_image[0]["img_path"])
img.show()

# Display citation details for the top matching image
print_text_to_image_citation(
    matching_results_image, print_top=True
)  # Print citation details for the top matching image

display_images(
    [
        matching_results_image[0]["img_path"],
        matching_results_image[1]["img_path"],
    ],
    resize_ratio=0.5,
)


# prompt = f""" Instructions: Compare the images and the Gemini extracted text provided as Context: to answer Question:
# Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.

# Context:
# Image_1: {matching_results_image_query_1[0]["image_object"]}
# gemini_extracted_text_1: {matching_results_image_query_1[0]["image_description"]}
# Image_2: {matching_results_image_query_1[1]["image_object"]}
# gemini_extracted_text_2: {matching_results_image_query_1[2]["image_description"]}

# Question:
#  - Key findings of Class A share?
#  - What are the critical differences between the graphs for Class A Share?
#  - What are the key findings of Class A shares concerning the S&P 500?
#  - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
#  - Identify key chart patterns in both graphs.
#  - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
# """

# # Generate Gemini response with streaming output

# ret = get_gemini_response(
#     multimodal_model,  # we are passing "gemini-2.5-flash" model
#     model_input=[prompt],
#     stream=True,
#     generation_config=GenerationConfig(temperature=1),
# )
