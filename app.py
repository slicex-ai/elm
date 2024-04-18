# Copyright (c) 2024, SliceX AI, Inc.

import streamlit as st
import json
import base64
import os
from elm.infer_elm_for_demo_app import load_elm_model_given_path, generate_elm_responses
st.set_page_config(layout="wide")
##########
##HEADER##
##########
LOGO_IMAGE = "logo.png"
st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:400 !important;
        font-size:20px !important;
        color: #fff !important;
        padding-top: 0px !important;
        padding-left: 10px !important;
    }
    .logo-img {
        float:right;
        margin-top: 7px !important;
        width: 20px;
        height: 20px
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">SliceX AI™ ELM (Efficient Language Models)</p>
    </div>
    """,
    unsafe_allow_html=True
)
with st.container(border=False, height=100):
         st.markdown(""":blue[ELM (family of Efficient Language Models) is the first version in the series of cutting-edge language models from <a href="https://slicex.ai/">:blue[SliceX AI]</a> that is designed to be modular and achieve the best in class performance in terms of quality, throughput & memory. This demo showcases a few ELM-v0.1 models (named Rambutan) from the first release. To learn more :point_right: <a href="https://medium.com/sujith-ravi/introducing-elm-efficient-customizable-privacy-preserving-llms-cea56e4f727d">:violet[ELM blog post]</a>]
         """, unsafe_allow_html=True)
##########
##HEADER##
##########

def get_preloaded_prompts(elm_model_path):
    if "news_classification" in elm_model_path:
        return "[INST]Below is a news article. Please classify it under one of the following classes (World, Business, Sports, Sci/Tech). Please format your response as a JSON payload.\n\n### Article: GM May Close Plant in Europe  DETROIT (Reuters) - General Motors Corp. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=GM.N target=/stocks/quickinfo/fullquote\"&gt;GM.N&lt;/A&gt; will likely  cut some jobs in Europe and may close a plant there as part of  a restructuring plan under development to try to return the  region to profitability, the U.S. automaker said on Wednesday.\n\n### JSON Response:[/INST]"
    elif "toxicity_detection" in elm_model_path:
        return "[INST]You are a helpful, precise, detailed, and concise artificial intelligence assistant. You are a very intelligent and sensitive, having a keen ability to discern whether or not a text message is toxic. You can also be trusted with following the instructions given to you precisely, without deviations.\nIn this task, you are asked to decide whether or not comment text is toxic.\nToxic content harbors negativity towards a person or a group, for instance:\n - stereotyping (especially using negative stereotypes)\n - disparaging a person's gender -- as in \"male\", \"female\", \"men\", \"women\"\n - derogatory language or slurs\n - racism -- as in discriminating toward people who are \"black\", \"white\"\n - cultural appropriation\n - mockery or ridicule\n - sexual objectification\n - homophobia -- bullying people who are \"homosexual\", \"gay\", \"lesbian\"\n - historical insensitivity\n - disrespecting religion -- as in \"christian\", \"jewish\", \"muslim\"\n - saying that certain groups are less worthy of respect\n - insensitivity to health conditions -- as in \"psychiatric/mental illness\"\n\nRead the comment text provided and predict whether or not the comment text is toxic. If comment text is toxic according to the instructions, then the answer is \"yes\" (return \"yes\"); otherwise, the answer is \"no\" (return \"no\").\nOutput the answer only as a \"yes\" or a \"no\"; do not provide explanations.\nPlease, never return empty output; always return a \"yes\" or a \"no\" answer.\nYou will be evaluated based on the following criteria: - The generated answer is always \"yes\" or \"no\" (never the empty string, \"\"). - The generated answer is correct for the comment text presented to you.\n### Comment Text: Dear Dr. Mereu, \n\n I am very much looking forward to this class. It is my first class at Rutgers! I think its extremely interesting and am very excited about it as I just decided that I want to minor in Psychology this year. I am especially interested in the neuroscience aspect of it all. Looking forward to a great semester!\n### Comment Text Is Toxic (Yes/No)  [/INST]"
    elif "news_content_generation" in elm_model_path:
        return "[INST]The following headline is the headline of a news report. Please write the content of the news passage based on only this headline.\n\nHeadline: Scientists Invent 'Invisible' Metamaterial With Bonus Reflect Mode \n\nContent:[/INST]"
    elif "news_summarization" in elm_model_path:
        return "[INST]You are given a news article below. Please summarize the article, including only its highlights.\n\n### Article: He is a World Cup winner, Spanish football legend, and one of the most recognisable faces in the classy Barcelona CF outfit. And now tourists are being offered the chance to walk in the footsteps of Andr\u00e9s Iniesta after he listed his beautiful Spanish vineyard on Airbnb. The world class midfielder took to Twitter to advise that his the 'Bodega Iniesta' vineyard he owns in Castilla-La Mancha can be rented out. Spain and Barcelona midfielder\u00a0Andr\u00e9s Iniesta is renting out his vineyard on Airbnb . Andr\u00e9s Iniesta posted on Twitter to spread the news that his charming vineyard can be rented out . And it's a snip of a price to tread paths made by one of the beautiful game's best players at \u20ac125 (\u00a390) a night. There is one bathroom and one bedroom at the charming little property, with other facilities including a kitchen, an indoor fireplace, Wi-Fi and parking provided. The residence is aimed for couples as the bedroom consists of one double bed. Decorated with a warm touch, guests can enjoy the fireplace with a glass of wine, relax on the couch or stroll among the vines. The vineyard stay comes with a guided tour of the area so you can get a real feel for the place . The interior of the property is simple yet subtle, ensuring the guests has all the amenities to get by . The house kitchen is fully equipped for people staying to use and enjoy. Breakfast food is provided for the duration of the stay, as well as towels and an extra set of sheets. Guests will also be advised of the surrounding area so they can explore for themselves. Also offered is a tour of the vineyard where guests can discover the secrets of wine-making. 'Airbnb gives you access to the most special places in the world',  Jeroen Merchiers, Regional Manager North, East and South of Europe told MailOnline Travel. The highlight of a stay at\u00a0Andr\u00e9s Iniesta's vineyard is undoubtedly what's outside rather than in . Guests can educate themselves in the art of wine-making, to hopefully produce their own special brand . 'Airbnb guests look for unique experiences. 'And we're pleased to announce Andr\u00e9s Iniesta is joining our community, unlocking a once in a lifetime experience for football and wine enthusiasts.' Some of the rules when staying in the property include being 'gentle with the vines,' smoking is prohibited inside, and the guests are responsible for goods during their stay. The property can be booked here. Iniesta has lit up the world of football for many years with his sublime skills, and now you can see a little more about his life outside the Beautiful Game . The 'Bodega Iniesta' vineyard he owns in Castilla-La Mancha can be rented out .\n\n### Summary:[/INST]"
    else:
        raise ValueError("Unsupported use-case")

def path_given_model_name(elm_model_path):
    return f"models/{elm_model_path}"


# model_list = [
#     "elm-1.0_news_classification",
#     "elm-1.0_toxicity_detection",
#     "elm-1.0_news_content_generation",
#     "elm-1.0_news_summarization",
#     "elm-0.75_news_classification",
#     "elm-0.75_toxicity_detection",
#     "elm-0.75_news_content_generation",
#     "elm-0.75_news_summarization",
#     "elm-0.25_news_classification",
#     "elm-0.25_toxicity_detection",
#     "elm-0.25_news_content_generation",
# ]

# selected_model = st.selectbox(
#                     'Choose a model',
#                     model_list, disabled=False)

use_case_list = [
    "news_classification",
    "news_content_generation",
    "toxicity_detection"
]

model_size_list = [
    "elm-1.0",
    "elm-0.75",
    "elm-0.25"
]

unsupported_model_list = set([
    "elm-0.25_news_summarization"
])

col1, col2 = st.columns(2)

selected_use_case = col1.selectbox(
                    'Choose use-case',
                    use_case_list, disabled=False) 

selected_model_size = col2.selectbox(
                    'Choose elm-model slice',
                    model_size_list, disabled=False) 
   
selected_model = f"{selected_model_size}_{selected_use_case}"
if selected_model in unsupported_model_list:
    st.write(f":orange[Combination of {selected_model_size} and {selected_use_case} is not added here. Please try another use-case.]")
    import sys; sys.exit()


if selected_model != st.session_state.get('selected_model', None):
    st.session_state["preloaded_prompt"] = get_preloaded_prompts(selected_model)
    st.session_state["model_info"] = load_elm_model_given_path(path_given_model_name(selected_model))
    prompt_config_file = os.path.join(path_given_model_name(selected_model), "example_prompts.json")
    with open(prompt_config_file, "r") as f:
        prompt_info = json.load(f)
    st.session_state['selected_model'] = selected_model
    st.write(f"Loaded: {selected_model}")


preloaded_prompt = st.session_state["preloaded_prompt"]
model_info = st.session_state["model_info"]

input_payload = st.text_area('Enter your Input (pre-populated with a sample prompt)', value=preloaded_prompt)
generate_button = st.button('Generate!')
if generate_button:
    #prompts = [prompt_info["template"].format(input=input_payload)]
    prompts = [input_payload]
    responses = generate_elm_responses(elm_model_path=path_given_model_name(selected_model), prompts=prompts, model_info=model_info, verbose=False)
    response = responses[0]
    with st.container(border=True, height=100):
        st.write(f":grey[ELM Response]:\n\n :blue[{response}]")
    print(response)

##########
##FOOTER##
##########
footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}
 a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
 .footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: #fff;
text-align: center;
}
</style>
<div class="footer">
<p> <a style='display: block; text-align: center;' href="https://slicex.ai/tos" target="_blank">Terms of Service</a>
© 2024 Copyright: SliceX AI, Inc. All Rights Reserved </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
##########
##FOOTER##
##########
