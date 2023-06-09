# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.11.3

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

# RUN git clone https://xxx:x-oauth-basic@github.com/KatayamaLab/simple-impedance-fitting.git

# Install production dependencies.
RUN pip3 install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
CMD streamlit run --server.port $PORT main.py