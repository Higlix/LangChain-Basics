import os
from dotenv import load_dotenv

try:
    from dotenv import load_dotenv

    load_dotenv()
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("API KEY MISSING")
except ImportError:
    print(ImportError)
    exit()


from langchain_anthropic import ChatAnthropic
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


anthropic_model = "claude-3-5-sonnet-latest"

file = open("article.txt", "r")
article = file.read()
file.close()


system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an assistant called {name} that helps generate catchy article titles",
    input_variables=["name"],
)

user_prompt = HumanMessagePromptTemplate.from_template(
      """You are tasked with creating a name for a article.
    The article is here for you to examine 
    
    ----
    
    {article}

    ----

    The name should be based of the context of the article.
    Be creative, but make sure the names are clear, catchy,
    and relevant to the theme of the article.

    Only output the article name, no other explanation or
    text can be provided.""",
    input_variables=["article"],
     
)

# basicly an fstring. inserts 'article' with the given value
# user_prompt.format(article="TEST STRING")

from langchain.prompts import ChatPromptTemplate

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])


# print(first_prompt.format(article="TEST STRING", name="JOE"))

llm = ChatAnthropic(temperature=0.0, model=anthropic_model)
creative_llm = ChatAnthropic(temperature=0.0, model=anthropic_model)

chain_one = (
    {
        "article": lambda x: x["article"],
        "name": lambda x: x["name"]
    }
    | first_prompt
    | creative_llm
    | {"article_title": lambda x: x.content}
)

article_title_msg = chain_one.invoke({
    "article": article,
    "name": "Joe"
})
print(article_title_msg)