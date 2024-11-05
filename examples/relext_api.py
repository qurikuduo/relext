# -*- coding: utf-8 -*-
"""
@author:dz
@description:
    # 实现三个 web api 接口，分别完成 article_triples_extract_demo.py 、 information_extract_demo.py 和 relation_extract_demo.py 的功能，\
    # 接口返回值的必备字段包括：状态码、消息、数据体，出错时数据体可以为空。
    # 使用uvicorn作为web api服务，使用 fastapi 作为前端服务。
"""
import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('..')
from relext import RelationExtraction
from relext import InformationExtraction
from relext import RelationExtraction


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArticleTriplesExtractRequest(BaseModel):
    art: str


class ArticleTriplesExtractResponse(BaseModel):
    code: int
    message: str
    result: dict
    art: str


class InformationExtractRequest(BaseModel):
    sch: list
    art: list


class InformationExtractResponse(BaseModel):
    code: int
    message: str
    result: list
    sch: list
    art: list


class RelationExtractRequest(BaseModel):
    art: str


class RelationExtractResponse(BaseModel):
    code: int
    message: str
    result: dict
    art: str



# 实现 article_triples_extract_demo.py 中功能的 web api 接口
@app.post("/v1/ArticleTriplesExtract", response_model=ArticleTriplesExtractResponse)
async def articleTriplesExtract(request: ArticleTriplesExtractRequest):
    #global model, tokenizer
    if len(request.art) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        art=request.art
    )

    logger.debug(f"==== request ====\n{gen_params}")

    # 增加异常捕获，捕获到异常后，打印异常，并将 code 设置为 500，message 为异常的堆栈信息
    try:
        triples = relationExtraction.extract(request.art)
        logger.debug(f"==== triples ====\n{triples}")
        # 定义 ArticleTriplesExtractResponse 的 resul 的类型为 triples 的类型
        response = ArticleTriplesExtractResponse(
            code=200,
            message="success",
            result=triples,
            art=request.art
        )

    except Exception as e:
        response = ArticleTriplesExtractResponse(
            code=500,
            message=str(e),
            result={},
            art=request.art
        )

    return response


# 实现 information_extract_demo.py 中功能的 web api 接口
@app.post("/v1/InformationExtract", response_model=InformationExtractResponse)
async def informationExtract(request: InformationExtractRequest):
    #global model, tokenizer
    if len(request.art) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        art=request.art,
        sch=request.sch
    )

    logger.debug(f"==== request ====\n{gen_params}")

    # 增加异常捕获，捕获到异常后，打印异常，并将 code 设置为 500，message 为异常的堆栈信息
    try:
        #sch = ['时间', '选手', '赛事名称']  # Define the sch for entity extraction
        #art = ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"]
        outputs = informationExtraction.extract(request.art, request.sch)

        logger.debug(f"==== outputs ====\n{outputs}")
        # 定义 ArticleTriplesExtractResponse 的 resul 的类型为 triples 的类型
        response = InformationExtractResponse(
            code=200,
            message="success",
            result=outputs,
            art=request.art,
            sch=request.sch
        )


    except Exception as e:
        response = InformationExtractResponse(
            code=500,
            message=str(e),
            result={},
            art=request.art,
            sch=request.sch
        )

    return response

# 实现 relation_extract_demo.py 中功能的 web api 接口
@app.post("/v1/RelationExtract", response_model=RelationExtractResponse)
async def relationExtract(request: RelationExtractRequest):
    if len(request.art) < 1:
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        art=request.art
    )

    logger.debug(f"==== request ====\n{gen_params}")

    # 增加异常捕获，捕获到异常后，打印异常，并将 code 设置为 500，message 为异常的堆栈信息
    try:
        triples = relationExtraction.extract(request.art)
        logger.debug(f"==== triples ====\n{triples}")
        # 定义 ArticleTriplesExtractResponse 的 resul 的类型为 triples 的类型
        response = RelationExtractResponse(
            code=200,
            message="success",
            result=triples,
            art=request.art
        )

    except Exception as e:
        response = RelationExtractResponse(
            code=500,
            message=str(e),
            result={},
            art=request.art
        )

    return response



if __name__ == "__main__":
    # 初始化 RelationExtraction
    relationExtraction = RelationExtraction()

    # 初始化 InformationExtraction
    informationExtraction = InformationExtraction()

    # 初始化 RelationExtraction
    #relationExtraction = RelationExtraction()

    #tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    #if 'cuda' in DEVICE:  # AMD, NVIDIA GPU can use Half Precision
    #model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
    # Multi-GPU support, use the following two lines instead of the above line, num gpus to your actual number of graphics cards
    #from utils import load_model_on_gpus
    #_my_gpu_count = torch.cuda.device_count()
    #print(f"GPU count: {_my_gpu_count}")
    #model = load_model_on_gpus(MODEL_PATH, num_gpus=_my_gpu_count)

    #else:  # CPU, Intel GPU and other GPU can use Float16 Precision Only
    #    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
    uvicorn.run(app, host='0.0.0.0', port=38000, workers=1)
