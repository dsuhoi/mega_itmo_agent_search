from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatModel

chooser_schemas = {
    "title": "itmo-info-chooser",
    "description": "Система оценки вопросов об университете ИТМО. Осуществляет переформулирование вопроса для поиска доп. информации в сети.",
    "type": "object",
    "properties": {
        "is_variants": {
            # сюда также можно добавить вариант с `0` для отсева нерелевантных вопросов
            "description": "1 - если вопрос содержит варианты ответа, 2 - если вопрос без вариантов ответ",
            "type": "integer",
            "enum": [1, 2],
        },
        "search_query": {
            "description": "Сформулируй содержательный текстовый запрос к поисковой системе (TavilyAPI) для получения необходимой информации для ответа на вопрос!",
            "type": "string",
        },
    },
    "required": ["is_correct", "search_query"],
}

assistant_schemas = {
    "title": "itmo-answers",
    "description": "Ты - гениальный знаток истории университета ИТМО! Ты знаешь всё об этом вузе и можешь дать точный ответ на поставленный вопрос из блока QUESTION! Ты можешь использовать информацию в блоках SEARCH_RESULTS для составления ответа",
    "type": "object",
    "properties": {
        "answer": {
            "description": "Номер правильного ответа на поставленный вопрос. Если вопрос не предполагает выбор из вариантов, то выведи null!",
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 10,
        },
        "reasoning": {
            "description": "Объяснение ответа или дополнительная информация по вопросу. Если в вопросе нет вариантов ответа, то здесь должен быть дан развернутый ответa! (не более 2-3 предложений)",
            "type": "string",
        },
        "sources": {
            "description": "Список номеров ссылок на используемые в ответе ресурсы из SEARCH_RESULTS. Если нет ссылок, то вернуть пустой array",
            "type": "array",
            "items": {
                "description": "Номер URL ссылки с используемой для ответа информацией",
                "type": "integer",
                "minimum": 1,
            },
        },
    },
    "required": ["answer", "reasoning", "sources"],
}


prompt = ChatPromptTemplate.from_messages(
    [("user", "QUESTION: {input}"), ("assistant", "SEARCH_RESULTS:\n{search_results}")]
)

prompt_chooser = ChatPromptTemplate.from_messages([("user", "QUESTION: {input}")])


def generate_chooser_agent(llm: BaseChatModel) -> BaseChatModel:
    """Создание агента для первичной валидации запроса + формулирование поисковых запросов к доп. источникам"""
    return (
        {"input": lambda x: x["input"]}
        | prompt_chooser
        | llm.with_structured_output(chooser_schemas)
    )


def generate_assistant_agent(llm: BaseChatModel) -> BaseChatModel:
    return (
        {
            "input": lambda x: x["input"],
            "search_results": lambda x: x["search_results"],
        }
        | prompt
        | llm.with_structured_output(assistant_schemas)
    )


def search_result_format(tavily_response: str, k: int = 5):
    """Форматирование результатов выдачи TavilyAPI"""
    res = ""
    for i, source in enumerate(tavily_response, start=1):
        if i > k:
            break
        res += f"{i}.) # {source['content']} #\n"
    return res + "###"


def create_assistant_algorithm(
    llm: BaseChatModel, model_name: str = "gpt-4o-mini", k: int = 5
):
    """Создание ассистента для ответов на вопросы об Университете ИТМО"""

    assistant_llm = generate_assistant_agent(llm)
    chooser_llm = generate_chooser_agent(llm)
    tavily_tool = TavilySearchResults(
        max_results=k,
        search_depth="advanced",
        include_domains=[
            "news.itmo.ru",
            "itmo.ru",
            "abit.itmo.ru",
            "edu.itmo.ru",
            "de.ifmo.ru",
            "itmo.events",
        ],
    )

    async def assistant_pipeline(query: str):
        """Агент для ответа на вопросы об Университете ИТМО"""

        response = await chooser_llm.ainvoke({"input": query})

        if response["is_variants"] in [1, 2]:
            search_response = await tavily_tool.ainvoke(response["search_query"])
            if search_response:
                search_res_str = search_result_format(search_response, k)
            else:
                search_res_str = "Информации из открытых источников нет! Требуется хорошо подумать и дать правильный ответ!"

            result = await assistant_llm.ainvoke(
                {"input": query, "search_results": search_res_str}
            )

            if result["sources"]:
                result["sources"] = [
                    search_response[i - 1]["url"] for i in result["sources"]
                ]

            if response["is_variants"] == 2:
                result["answer"] = None

        else:  # для варианта с отсевом нерелевантной информации
            result = {
                "answer": None,
                "reasoning": "Вопрос не связан с тематикой сервиса!",
                "sources": [],
            }

        result["reasoning"] += f"\nОтвет подготовлен моделью {model_name}."
        return result

    return assistant_pipeline
