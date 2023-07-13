import json
import os
import time

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

from third_party import linkedin

experience_prompt = """
Read the description of someones accomplishments from a job inside the triple
backticks below. Create a JSON formatted list of the accomplishments in the
description.

```
{description}
```

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
        input_variables=["description"], template=experience_prompt
    )

    with open("ericness.json", "r") as fp:
        profile = json.load(fp)
    for experience in profile["experiences"]:
        chain = LLMChain(llm=llm, prompt=experience_prompt_template)
        print(chain.run(description=experience["description"]))
        time.sleep(1)
