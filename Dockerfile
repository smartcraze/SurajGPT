FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

COPY . .

ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}


RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt



RUN python -c "from rag_chain import setup_vectorstore; setup_vectorstore()"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
