# MEGA ITMO Agent Search Service
Система для предоставления ответов на вопросы об Университете ИТМО.

## Сборка
Создайте файл `.env` в корневой директории проекта со следующим содержимым:
```bash
OPENAI_API_KEY="sk-proj-..."
TAVILY_API_KEY="tvly-..."
MODEL_NAME="gpt-4o-mini"
```
([Информация о TAVILY_API_KEY](https://app.tavily.com/home))

Для запуска и загрузки контейнера выполните команду:

```bash
docker-compose up -d --build
```
Она соберёт Docker-образ, а затем запустит контейнер.

После успешного запуска контейнера приложение будет доступно на `http://localhost:10101`.

## Проверка работы
Отправьте POST-запрос на эндпоинт `/api/request`. Например, используйте curl:

```bash
curl --location --request POST 'http://localhost:10101/api/request' \
--header 'Content-Type: application/json' \
--data-raw '{
  "query": "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
  "id": 1
}'
```
В ответ вы получите JSON вида:

```json
{
  "id": 1,
  "answer": 2,
  "reasoning": "Из информации на сайте",
  "sources": [
    "https://itmo.ru/ru/",
    "https://abit.itmo.ru/"
  ]
}
```
Выходные данные представляются в следующих полях:
- `id` будет соответствовать тому, что вы отправили в запросе,
- `answer` соответствует варианту правильного ответа (или `null`, если не указаны варианты),
- `reasoning` содержит информацию о деталях ответа или другую информацию, связанную с вопросом,
- `source` содержит список ссылок на сайты с информацией, задействованной в ответе.

## Доп. информация
Чтобы остановить сервис, выполните:

```bash
docker-compose down
```
