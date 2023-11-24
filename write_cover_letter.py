import json
import os
import pprint
import time

from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from third_party import linkedin

EXPERIENCE_PROMPT = """
Read the description of someones accomplishments from a job inside the triple
backticks below. Create a JSON formatted list of the accomplishments in the
description.

```
{description}
```
"""

JOB_PROMPT = """
Read the job post inside the triple backticks below. Identify the responsibilities
and qualifications for the job. These are both typically in a list of bullet
points. Return a JSON record with the list of the responsibilities under one
key and the list of qualifications under the other.

```
{job_description}
```

"""

LETTER_PROMPT = """
You are a candidate for a job. You are well-qualified, professional and
excellent at communication and persuasion. Write a cover letter for the
job described below. The qualifications and responsibilities are listed
in two separate sections in triple backticks.

```
Responsibilities: {responsibilities}
```

```
Qualifications: {qualifications}
```

First decide which responsibilities are most important to address in
the cover letter. These are usually related to specific skills or
technologies. Pick one or two to address in the cover letter. Print
the ones you select before writing the letter.

Second decide which qualifications are most important to address in
the cover letter. The more difficult to obtain the qualification
the better it is to include in the cover letter. 
Pick one or two to address in the cover letter. Print
the ones you select before writing the letter.

Now match your accomplishments to the responsibilities and qualifications
that you selected. You accomplishments are available to retrieve from
the attached vector store. Print out the accomplishment that you
will match with each responsibility and qualification.

Finally write your letter. Be sure to include the accomplishments that
you've matched to the responsibilities and qualifications.
"""

if __name__ == "__main__":
    # result = linkedin.scrape_linkedin_profile(
    #     "https://www.linkedin.com/in/ericnessdata/"
    # )
    # with open("ericness.json", "w") as fp:
    #     json.dump(result, fp)
    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        verbose=True,
    )
    experience_prompt_template = PromptTemplate(
        input_variables=["description"], template=EXPERIENCE_PROMPT
    )

    with open("ericness.json", "r") as fp:
        profile = json.load(fp)
    documents = []
    chain = LLMChain(llm=llm, prompt=experience_prompt_template)
    for experience in profile["experiences"][:3]:
        accomps = chain.run(description=experience["description"])
        for accomp in json.loads(accomps)["accomplishments"]:
            documents.append(
                Document(
                    page_content=str(
                        {
                            "starts_at": experience["starts_at"],
                            "ends_at": experience["ends_at"],
                            "company": experience["company"],
                            "title": experience["title"],
                            "accomplishment": accomp,
                        }
                    )
                )
            )
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=db.as_retriever()
    # )
    # query = "What kind of pricing models has the candidate worked on?"
    # answer = qa.run(query)
    # print(answer)

    # job_url = "https://www.linkedin.com/jobs/view/3407202764"
    # job = linkedin.scrape_linkedin_job(job_url)
    # with open("tiktok_job.json", "w") as fp:
    #    json.dump(job, fp)

    with open("tiktok_job.json", "r") as fp:
        job = json.load(fp)
    job_prompt_template = PromptTemplate(
        input_variables=["job_description"], template=JOB_PROMPT
    )
    job_chain = LLMChain(llm=llm, prompt=job_prompt_template)
    job_result_raw = job_chain.run(job_description=job["job_description"])
    job_result = json.loads(job_result_raw)

    # tools = [
    #     Tool(
    #         name="Search",
    #         func=db.similarity_search,
    #         description="useful when you need to access candidate accomplishments",
    #     ),
    # ]
    # react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
    # react.run()

    letter_prompt_template = PromptTemplate(
        input_variables=["responsibilities", "qualifications"], template=LETTER_PROMPT
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": letter_prompt_template},
    )
    result = qa_chain(
        {
            "responsibilities": job_result["responsibilities"],
            "qualifications": job_result["qualifications"],
        }
    )
    print(result["result"])
