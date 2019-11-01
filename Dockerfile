FROM python:3.6

WORKDIR /policyagent

## update pip to get tensorflow 2.0.0
RUN pip install --upgrade pip

## copy sourcecode
ADD policygradient policygradient
ADD agentapi agentapi
ADD setup.py setup.py
ADD gunicorn.config gunicorn.config

## install package and gunicorn
RUN pip install .
RUN pip install gunicorn

## expose port
EXPOSE 8080

## run the server
CMD ["gunicorn", "-c", "gunicorn.config", "agentapi.api"]
