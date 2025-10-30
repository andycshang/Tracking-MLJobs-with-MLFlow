# set Python environment
FROM python:3.9-slim

#set work directory
WORKDIR /A0

#copy file into container
COPY . /A0

#install requirements
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 8000

# do
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

