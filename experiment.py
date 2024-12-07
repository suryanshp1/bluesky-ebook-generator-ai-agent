from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import Optional
from langchain_groq import ChatGroq
from atproto import Client, models
from cloudinary.uploader import upload
import cloudinary
import traceback
import logging
import os
from dotenv import load_dotenv
import datetime
from fpdf import FPDF, HTMLMixin
from fpdf.enums import XPos, YPos
import gc
import time
import json
import markdown2
from docx import Document


