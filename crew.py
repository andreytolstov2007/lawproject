import os
import telebot
import logging
from telebot import types
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from crewai_tools import (
    DirectoryReadTool#,
#    WebsiteSearchTool
)
from dotenv import load_dotenv, dotenv_values
from crewai_tools import BrightDataSearchTool

# read file tool imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from transformers import pipeline 
import sys 

sys.setrecursionlimit(3000)

# load environment vars and log config
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
bot_token = os.getenv("BOT_TOKEN")
os.environ["BRIGHT_DATA_API_KEY"] = os.getenv("BRIGHT_DATA_API_KEY")
os.environ["BRIGHT_DATA_ZONE"] = os.getenv("BRIGHT_DATA_ZONE")


# --- 1. Telegram Bot Setup ---
# Initialize the Telegram bot with your token from the .env file
bot = telebot.TeleBot(bot_token)


# --- 2. CrewAI Setup ---
# Initialize the language model (LLM)
############ LLM ########### 
# Define LLM
llm = LLM(
    #model="Yandex",    # good
    #model="Vikhr",     # низкое качество ответа
    #model="MistralNemo", # 
    #model="Mistral", # не дает точных ответов
    model="MistralQ4", # !!! very good
    #model="Llama38", # function calling? долго работает
    #model="GigaChat", # долго
    #model="Llama2", # нет ответа
    #model="mLlama",
    base_url="http://localhost:11434/v1",
    temperature = 0.5
)

# --- 3. Instantiate tools ---
directory_tool = DirectoryReadTool(directory='data')
#web_rag_tool = WebsiteSearchTool("https://mosdigitals.ru/blog/chastye-voprosy-yuristu")
############ Tools ########### 
# Tool 1 - FAQ
@tool("lookup_law")
def lookup_law_info(question: str) -> str:
    """
    This function processes a user's question and returns an appropriate answer.
    """
    question = question.lower()
    print("####")
    if "привет" in question:
        return "Привет. Чем могу помочь?"
    elif "кража шоколадки" in question:
        return "Хищение имущества стоимостью до 1000 рублей — мелкое хищение по ч. 1 ст. 7.27 КоАП РФ: штраф до пятикратной стоимости похищенного"    
    elif "нарушение миграционного законодательства" in question:
        return '''предусмотрены штрафы и возможное административное выдворение по главе 18 КоАП РФ, в том числе ст. 18.8, 18.9 КоАП РФ. 
        За организацию незаконной миграции действует уголовная ответственность по ст. 322.1 УК РФ'''         
    elif "what is the fine for exceeding the speed limit by 20 km/h?" in question:
        return "1000 rubles"
    else:
        return "i'm sorry, i don't have an answer for that question at the moment"

# Tool 2 - Search Web
@tool("search_web")
def search_web_info(question: str) -> str:
    """
    This function processes a user's question and returns an appropriate answer.
    """    
    tool = BrightDataSearchTool(
        query=question,
    )
    return tool.run()  

# Tool 3 - Search Files
@tool("search_files")
def search_file_info(question: str) -> str:
    """
    This function processes a user's question and returns an appropriate answer.
    """        
    folder_path = "data"   
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load()) 
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents) 
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()
    print('***') 
    llm_pipeline = pipeline("text-generation", model="gpt2-large", device=0, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    print('****')
    prompt_template = "Answer the following question based on the provided context: {context}\n\nQuestion: {query}\nAnswer:"
    prompt = PromptTemplate(input_variables=["query", "context"], template=prompt_template)
    chain = prompt | llm
    #chain.invoke(input={"bookname": "The name of the rose"})
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    #query = "Какой штраф за проезд на запрещающий сигнал светофора?"
    query = question
    #print(retriever)
    # IMPORTANT: using only the top-1 document by default
    retrieved_docs = retriever.invoke(query)[:1]
    context = " ".join([doc.page_content for doc in retrieved_docs])
    context = truncate_to_max_tokens(context, max_tokens=500)
    response = retrieval_qa.invoke(query)
    print("Answer:", response)   
    return response


############ 4. Agents ########### 
# Agent respondent
respondent = Agent(
    role="Респондент",
    goal='''Предоставлять ясные, точные и исчерпывающие ответы на вопросы пользователей, чтобы облегчить их взаимодействие с сервисом, 
            повысить удовлетворённость клиентов и минимизировать количество повторных обращений. ''',
    backstory='''Специалист по коммуникациям и клиентскому обслуживанию с многолетним опытом работы в сфере поддержки пользователей и 
        предоставления консультаций по продуктам и услугам. Известна своим дружелюбным характером, терпением и готовностью подробно 
        разъяснять любые вопросы клиентам. Владеет искусством упрощённого объяснения технических деталей, легко адаптируется к 
        различным типам аудитории и стремится обеспечить высокое качество обслуживания каждого обратившегося пользователя.''',
    llm=llm,
    verbose=True, # Set to True for detailed logs in your console
    memory=True, # Enables short-term, long-term, and entity memory
    allow_delegation=True, # Enables asking questions to other agents
    tools = [lookup_law_info],
    #tools=[lookup_law_info,search_web_info,search_file_info],
    #tools=[search_web_info],
    max_iter=3
)

# Agent researcher
researcher = Agent(
    role="Юридический исследователь",
    goal="Изучение и анализ правовых актов, судебной практики и нормативных документов. Извлечение ключевой информации.",
    backstory='''Юридический исследователь высокого класса, специализирующийся на изучении правовых актов, судебной практики
        и нормативных документов федерального и регионального уровней. Отличается острым умом, глубокой эрудицией и 
        стремлением докопаться до сути любого вопроса. Его отличает высокий профессионализм, критическое мышление6
        и способность систематизировать огромные массивы информации.''',
    llm=llm,
    verbose=True, # Set to True for detailed logs in your console
    memory=True, # Enables short-term, long-term, and entity memory
    allow_delegation=False, # Enables asking questions to other agents
    #tools=[directory_tool],
    #tools=[search_web_info,search_file_info],
    #tools=[search_web_info],
    tools=[search_web_info,search_file_info],
    max_iter=2
)

# Agent consultant
consultant = Agent(
    role="Консультант по правовым вопросам",
    goal="Интерпретация законов и постановлений. Предоставить рекомендации и разъяснения по правовым вопросам",
    backstory='''Юрист-консультант международного уровня с глубокими познаниями российского права, гражданского кодекса, 
        корпоративного управления и коммерческой практики. Обладает высоким уровнем эмпатии и способностью быстро 
        адаптироваться к новым ситуациям. Отличают высокая ответственность, внимание к деталям и способность находить 
        решения даже в сложных правовых ситуациях.''',
    llm=llm,
    verbose=True, # Set to True for detailed logs in your console
    memory=True, # Enables short-term, long-term, and entity memory
    allow_delegation=True, # Enables asking questions to other agents
    max_iter=2 # maximum number of iterations an agent can perform to execute a single task before it is forced to provide a final answer or stop. 
)

# Agent editor
editor = Agent(
    role="Редактор и переводчик.",
    goal='''Организация и представление результатов исследования по правовым вопросам. 
        Обеспечить максимальную доступность и удобство восприятия информации пользователями.''',
    backstory='''Высококвалифицированный редактор, организатор и переводчик с богатым опытом работы в области права и юриспруденции. 
        Занимает лидирующие позиции среди коллег, демонстрируя глубокое понимание предмета и отличные навыки управления процессами 
        подготовки материалов. Обладает превосходными лингвистическими навыками и способна подготовить документацию на высочайшем уровне, 
        сохраняя точность и стиль исходного текста.''',
    llm=llm,
    verbose=True, # Set to True for detailed logs in your console
    memory=True, # Enables short-term, long-term, and entity memory
    allow_delegation=False, # Enables asking questions to other agents
    max_iter=2
)

# 5. Define a function to create a task and run the crew
def create_tasks(topic):

    respondent_task = Task(
        description=f"Ответь, используя FAQ: {topic}. Используй историю для контекста. Если вопроса нет в FAQ передай таску '{researcher.role}' для исследования или '{consultant.role}' для консультации. Если в вопросе не указан регион, то формируем ответ для России. Сформируй ответ на языке запроса.",
        expected_output="FAQ ответ или делегирование таски.",
        agent=respondent,
        verbose=True
        #tracing=True#,
        #human_input=True
    ) 

    researcher_task = Task(
        description=f"Ответь на вопрос: {topic}. Изучи российские нормативные документы и предоставь ключевую информацию по вопросу. Сформируй ответ на языке запроса.",
        expected_output="Ключевая информация по нормативным документам.",
        agent=researcher,
        verbose=True
        #tracing=True
    )

    consultant_task = Task(
        description=f"Ответь на вопрос: {topic}. Предоставь инструкции по правовому вопросу. Если нужен глубокий анализ, передай таску {researcher.role}. Сформируй ответ на языке запроса.",
        expected_output="Подробная информация и инструкции по правовой ситуации.",
        agent=consultant,
        verbose=True
        #tracing=True
    )   

    editor_task = Task(
        #description=f"Ответь на вопрос: {topic}. Суммируй информацию команды по вопросу. Переведи на русский язык, если вопрос на русском языке, иначе переведи на английский.",
        description=f"Ответь на вопрос: {topic}. Объедини информацию команды по вопросу. Переведи на русский язык, если вопрос на русском языке, иначе переведи на английский.",
        expected_output="Финальный ответ.",
        agent=editor,
        verbose=True
        #tracing=True
    )      
    return [respondent_task, researcher_task, consultant_task, editor_task]
    #return [researcher_task]

### 6. Run multi agent
def run_multi_agent(user_input):
#def run_multi_agent(inputList):
    crew = Crew(
        #agents=[respondent, researcher, consultant, editor],
        agents=[researcher],
        tasks=create_tasks(user_input),
        #tasks=[respondent_task, researcher_task, consultant_task, editor_task],
        process=Process.sequential,
        prompt_file="custom_prompts.json",
        verbose=True
        #tracing=True
    )
    
    result = crew.kickoff(inputs={'topic': user_input})
    #result = crew.kickoff_for_each(inputs=inputList
    #     # Specifies that each list item goes into the '{topic}' variable
    #)
    #result = crew.kickoff_for_each(inputs=[{'topic': 'Какое наказание за мелкое хулиганство?'}])

    return result

# --- 7. Telegram Message Handlers ---
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Hello! Send me a question, and I will use my AI agents to answer it.")

@bot.message_handler(content_types=['text'])
def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text
    
    # Send a "typing" action and confirmation message
    bot.send_chat_action(chat_id, 'typing')
    bot.send_message(chat_id, f"Processing: '{user_text}'...")
    
    try:
        # Pass the user's message to the CrewAI system
        response = run_multi_agent(user_text)
        bot.send_message(chat_id, f"{response}", parse_mode='Markdown')
    except Exception as e:
        bot.send_message(chat_id, f"An error occurred: {e}")

# --- 8. Test function
def runTest():
    query_list = []
    response_list = []
    #query_list.append("Какое наказание за превышение скорости на 30 км/ч?")
    #query_list.append("Какое наказание за вождение в состоянии опьянения?")
    #query_list.append("Какое наказание за оставление места ДТП?")
    #query_list.append("Какое наказание за управление ТС без водительских прав?")
    #query_list.append("Какое наказание за распитие алкоголя в общественных местах?")
    #query_list.append("Какое наказание за мелкое хулиганство?")
    #query_list.append("Какое наказание за кражу шоколадки за 100 рублей?")
    #query_list.append("Какое наказание за продажу алкоголя без лицензии?")
    #query_list.append("Какое наказание за нарушение миграционного законодательства?")
    query_list.append("Какое наказание за нарушение трудового законодательства?")
    
    for i in range(0,len(query_list)):
        print("Processing Q"+str(i+1)+". "+query_list[i])
        response = run_multi_agent(query_list[i])
        print(response)

# --- 4. Start the Bot ---
if __name__ == '__main__':
    #runTest()
    print("Bot is polling...")
    bot.infinity_polling()