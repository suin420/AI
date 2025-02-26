# AI Backend Server

## Setup

### 1. Install Python

### 2. install modules of Python

```
pip3 install fastapi
pip3 install "uvicorn[standard]"
```

## How To Run API Server

### Run API Server

```
> uvicorn api_server:app --host 0.0.0.0 --port 9000
// if you want to change port number, modify value of `--port`
```

## Documentation

- Use Open API Document (Swagger Page)
  - During server is running, Access `https://127.0.0.1:9000/docs` in Web Browser.
  - you can see the documents of API and you can also test it.
