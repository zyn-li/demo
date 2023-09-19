from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

load_dotenv()

loader = CSVLoader(file_path="dataset_400.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=2)
    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
template = """
    You are a top class IRS investigator.
    I will share a summary of criminal case with you, the summary includes some key informations including:
    The suspect, This crimial case happend in, This suspect's criminal activity is, This result of this case, The amount of money involved.
    You will help me to write the best press release article that I should use based on past best practies, and you will following ALL of the rules below:
    1. The press release article you write should be very similar, in terms of length, ton of voice, logical arguments and other details.
    2. If the best practice are irrelevant, then try to mimic the style of the best practice to the title
    Below is summary  I have:
    {summary}
    Here is a list of best practies of how we normally write press release articles based on the summary:
    {best_practices}
    Also, here are some additional information you need to know:
    1. If you need to mention the name of an investigator, remember his name can not be same as the suspect.
    2. Usually the amount of money is given in the summary, but if you don't know how much money is invovled, you shold come up with a random amount within a reasonable range to use as a placeholder. You cannot say the exact amount of money involved can not be determined.
    Please write the best press release that I should use given the summary I have now.
"""

promt = PromptTemplate(
    input_variables=['summary', 'best_practices'],
    template=template
)

chain = LLMChain(llm=llm, prompt=promt)


def generate_response(summary):
    best_practices = retrieve_info(summary)
    response = chain.run(summary=summary, best_practices=best_practices)
    return response


def main():
    header_image = Image.open('header.png')

    st.set_page_config(
        page_title="IRS Press Release Generator",
    )
    st.image(header_image)
    st.header("Press Release Generator Demo")

    if not st.session_state.get("submitted"):
        with st.form("summary"):
            suspect = st.text_input("The suspect is...")
            case = st.text_input("This criminal case happened in...")
            activity = st.text_input("This suspect's criminal activities are...")
            result = st.text_input("The result of this case...")
            amount = st.text_input("The amount of money involved...")
            if st.form_submit_button("Submit"):

                st.session_state['submitted'] = True

    if st.session_state.get('submitted'):
        if suspect and case and activity and result and amount:
            summary = "\n".join([suspect, case, activity, result, amount])
            st.write(f"Your Summary is \n: {summary}")
            st.write("Generating Response, it may take some time...")
            response = generate_response(summary)
            st.info(response)
            st.session_state['submitted'] = False
        else:
            st.write("Missing Information...")
            st.session_state['submitted'] = False


if __name__ == "__main__":
    main()




