from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from atproto import Client, models
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
import cloudinary
import traceback
import logging
import os
from dotenv import load_dotenv
import datetime
from fpdf import FPDF
import gc
import time

load_dotenv()

# Set up the logger
logging.basicConfig(
    filename='app.log',  # log file name
    level=logging.INFO,  # log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger instance
logger = logging.getLogger(__name__)

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    BLUESKY_USERNAME = os.getenv('BLUESKY_USERNAME')
    BLUESKY_PASSWORD = os.getenv('BLUESKY_PASSWORD')
    EBOOK_OUTPUT_DIR = 'ebooks'
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')

client = Client()
client.login(Config.BLUESKY_USERNAME, Config.BLUESKY_PASSWORD)

cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)


def upload_to_cloudinary(file_path: str, topic: str) -> tuple:
    """
    Upload a file to Cloudinary and return the URL and public_id
    """
    try:
        # Create a folder name from the topic
        folder_name = "ebooks"
        # Create a public_id from the topic (removing spaces and special characters)
        public_id = f"{folder_name}/{topic.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Upload the file
        result = upload(
            file_path,
            resource_type="raw",  # For PDF files
            public_id=public_id,
            folder=folder_name,
            tags=[topic, "ebook"],
            overwrite=True,
            access_mode="public"
        )
        
        # Get the secure URL
        url = result['secure_url']
        public_id = result['public_id']
        
        logger.info(f"File uploaded successfully to Cloudinary. URL: {url}")
        return url, public_id
        
    except Exception as e:
        logger.exception(f"Error uploading to Cloudinary: {str(e)}")
        raise e

def delete_from_cloudinary(public_id: str):
    """
    Delete a file from Cloudinary using its public_id
    """
    try:
        result = cloudinary.api.delete_resources([public_id], resource_type="raw")
        logger.info(f"File deleted from Cloudinary: {result}")
    except Exception as e:
        logger.exception(f"Error deleting from Cloudinary: {str(e)}")

class EditorAgent:
    def __init__(self):
        self.prompt = PromptTemplate(input_variables=["content"],
                                     template="""You are an expert eBook content editor. Your task is to review and correct the given chapter content to improve its overall quality. Ensure that the content is clear, well-structured, grammatically correct, and written in a professional yet approachable tone.

        Here are your instructions:

        1. **Grammar and Spelling**: Correct any grammatical errors, spelling mistakes, or awkward phrasing.
        2. **Clarity**: Ensure the content is easy to understand. Simplify complex sentences where necessary and break down any overly complicated ideas.
        3. **Structure**: Ensure that the chapter has a logical flow with proper transitions between paragraphs. Add or adjust headings, subheadings, and sections if needed for better readability.
        4. **Tone**: Ensure the writing has a formal yet accessible tone, suitable for a general audience, such as students, researchers, or general readers.
        5. **Conciseness**: Eliminate redundancy and unnecessary filler content while keeping the chapter comprehensive.
        6. **Consistency**: Maintain consistent terminology, style, and formatting throughout the content.
        7. **Focus**: Ensure that the content stays on topic and avoids unnecessary digressions.

        Input:
        - **Chapter Content**: {content}

        Now, review and correct the given chapter content based on the instructions. Return the improved content as a single corrected string of text.
        Note: Don't include chapter number and chapter title in the content.
        \n\n placeholder:{agent_scratchpad}
        """)
        
        self.llm = ChatGroq(api_key=Config.GROQ_API_KEY, model="Llama-3.1-8b-instant")
        self.tools = []

    def edit(self, content):
        try:
            # Use the prompt to edit chapter content
            research_agent = create_tool_calling_agent(self.llm,self.tools,self.prompt)
            research_agent_executor = AgentExecutor(agent=research_agent, tools=self.tools)
            response = research_agent_executor.invoke({"content":content})
            edited_content = response.get("output")

            return edited_content

        except Exception as e:
            logger.exception(F"Error: {e} | Traceback: {traceback.format_exc()}")
            raise e
        
class ResearcherAgent:
    def __init__(self):
        self.prompt = PromptTemplate(input_variables=["topic"],
                                     template="""Generate a detailed outline with exactly 12 chapter titles on the given topic: {topic} . \noutput format:\n
            Example: ['Title1', 'Title2', 'Title3', 'Title4', 'Title5', 'Title6', 'Title7', 'Title8', 'Title9', 'Title10', 'Title11', 'Title12']\n
            Ensure that the output contains exactly 12 chapters and that the chapter titles do not contain any special keywords or symbols. \nNote: Strictly output should be only a proper python list like ['Title1', 'Title2', 'Title3', 'Title4', 'Title5', 'Title6', 'Title7', 'Title8', 'Title9', 'Title10', 'Title11', 'Title12']. **Not a single world is allowed except the title list in the output**
            Ignore below text:
            \n placeholder:{agent_scratchpad}""")
        
        self.llm = ChatGroq(api_key=Config.GROQ_API_KEY, model="Llama-3.1-8b-instant")
        # self.tools = [search]
        self.tools = []

    def research(self, topic):
        try:
            # Use the prompt to generate chapter content
            research_agent = create_tool_calling_agent(self.llm,self.tools,self.prompt)
            research_agent_executor = AgentExecutor(agent=research_agent, tools=self.tools)
            response = research_agent_executor.invoke({"topic":topic})
            response = response.get("output")

            chapters = [a.strip("'").strip(" ").strip("\n").strip("'") for a in response[1:-1].split(", ")]

            return chapters

        except Exception as e:
            logger.exception(F"Error: {e} | Traceback: {traceback.format_exc()}")
            raise e
        
class WriterAgent:
    def __init__(self):
        self.prompt = PromptTemplate(input_variables=["topic", "chapter"],
                                     template="""You are an expert eBook writer. Your task is to write a detailed, well-structured, and engaging chapter for an eBook based on the given chapter title and overall topic of the book. The chapter should be informative, easy to understand, and provide in-depth coverage of the subject matter.

        Here are your instructions:

        1. The content should be structured with proper headings, subheadings, and sections.
        2. The writing should be in a clear and engaging style, suitable for a broad audience such as students, researchers, or curious readers.
        3. Ensure the chapter content stays focused on the chapter title, but it should also be aligned with the overall topic of the book.
        4. Provide relevant examples, explanations, and any necessary background information to make the content more comprehensive and easier to understand.
        5. The length of the chapter should be around 1500-2000 words.
        6. Use a formal yet approachable tone. Avoid overly technical jargon unless it's explained well.

        Inputs:
        - **Topic**: {topic}
        - **Chapter Title**: {chapter}

        Now, based on the provided inputs, write the complete content for this chapter. The content should be returned as a single string of text.

        Note: Don't include chapter number and chapter title in the content.
        \n\n placeholder:{agent_scratchpad}
        """)
        
        self.llm = ChatGroq(api_key=Config.GROQ_API_KEY, model="Llama-3.1-8b-instant")
        self.tools = []


    def write(self, chapter, topic):
        try:
            # Use the prompt to generate chapter titles
            research_agent = create_tool_calling_agent(self.llm,self.tools,self.prompt)
            research_agent_executor = AgentExecutor(agent=research_agent, tools=self.tools)
            response = research_agent_executor.invoke({"topic":topic, "chapter": chapter})
            chapter_content = response.get("output")

            return chapter_content

        except Exception as e:
            logger.exception(F"Error: {e} | Traceback: {traceback.format_exc()}")
            raise e
        
def generate_pdf(topic, chapters_content, chapters):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Cover Page
        pdf.add_page()
        pdf.set_font(family='Times', size=30)
        pdf.set_text_color(r=0, g=0, b=0)
        pdf.set_y(pdf.h / 2 - 15)
        pdf.cell(w=pdf.w - 50, txt=f"{topic}", align="C")

        # Table of Contents
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Table of Contents", ln=True)
        for i, chapter_title in enumerate(chapters):
            pdf.cell(200, 10, txt=f"Chapter {i+1}: {chapter_title}", ln=True)

        # Chapters
        for i, content in enumerate(chapters_content):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt=f"Chapter {i+1}: {chapters[i]}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, txt=content, markdown=True)
        
        # Ensure output directory exists
        os.makedirs(Config.EBOOK_OUTPUT_DIR, exist_ok=True)

        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file_path = f"{Config.EBOOK_OUTPUT_DIR}/{topic.replace(' ', '_')}_{timestamp}.pdf"

        pdf.output(pdf_file_path)

        return pdf_file_path

    except Exception as e:
        logger.exception(F"Error: {e} | Traceback: {traceback.format_exc()}")
        raise e
        
class EbookGenerator:
    
    def generate_ebook_task(self, topic): 
        try:
            if not topic:
                raise Exception("Topic not provided")
            
            # Initialize agents
            logger.info("Initializing agents...")
            researcher = ResearcherAgent()
            writer = WriterAgent()
            editor = EditorAgent()

            # Step 1: Research the topic and get chapter titles
            logger.info("Researching the topic...")
            chapters = researcher.research(topic)

            # Step 2: Write content for each chapter
            logger.info("Writing content for each chapter...")
            written_content = [writer.write(chapter, topic) for chapter in chapters]

            # Step 3: Edit each chapter for grammar and consistency
            logger.info("Editing each chapter for grammar and consistency...")
            final_content = [editor.edit(content) for content in written_content]

            # Step 4: Decoding text to latin-1
            logger.info("Decoding text to latin-1...")
            final_content = [content.encode('latin-1', 'ignore').decode('latin-1') for content in final_content]

            # Step 5: Generate PDF with all content
            logger.info("Generating PDF with all content...")
            pdf_file_path = generate_pdf(topic, final_content, chapters)

            # Return PDF file path
            logger.info("Task completed.")

            collected = gc.collect()

            logger.info(f"Garbage collector: collected {collected} objects.")
            return pdf_file_path

        except Exception as e:
            logger.exception(F"Error: {e} | Traceback: {traceback.format_exc()}")
            raise e

def reply_with_pdf(mention: dict):
    """Reply to a mention with a PDF eBook."""
    try:
        logger.info("Processing mention reply_with_pdf ...")
        # Extract topic from the mention text
        text = mention.record.text
        words = text.split()
        topic = " ".join(words[1:])  # Assuming the topic is after the bot mention
        logger.info(f"Topic: {topic}")

        # Generate content and create PDF
        pdf_path = EbookGenerator().generate_ebook_task(topic)
        # pdf_path = f"{Config.EBOOK_OUTPUT_DIR}/Yoga.pdf"
        logger.info(f"PDF path: {pdf_path}")

        # Upload PDF to Cloudinary
        download_url, public_id = upload_to_cloudinary(pdf_path, topic)
        logger.info(f"PDF uploaded to: {download_url}")
        
        try:
            # Create proper reply reference
            reply_ref = {
                "root": {
                    "cid": mention.cid,
                    "uri": mention.uri
                },
                "parent": {
                    "cid": mention.cid,
                    "uri": mention.uri
                }
            }
            
            # Create the embed external object with the download link
            embed = models.AppBskyEmbedExternal.Main(
                external=models.AppBskyEmbedExternal.External(
                    title=f"Ebook: {topic}",
                    description="Click to download your generated ebook",
                    uri=download_url,
                    thumb=None
                )
            )
            
            # Reply to the mention with the download link
            client.post(
                text=f"🤖 Here's your ebook about {topic}! 📚\nClick here to download: {download_url}",
                reply_to=reply_ref,
                embed=embed
            )
            
            logger.info(f"PDF link sent successfully!")
            print("PDF link sent successfully!")
            
            # Clean up local file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"Local PDF file cleaned up: {pdf_path}")
                
        except Exception as e:
            # If posting fails, clean up the uploaded file from Cloudinary
            delete_from_cloudinary(public_id)
            raise e
        
    except Exception as e:
        logger.exception(f"Error in reply_with_pdf: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    """Main bot loop to monitor and reply to mentions."""
    print("Bot started. Listening for mentions...")
    # Keep track of processed notifications
    processed_notifications = set()

    while True:
        notifications = client.app.bsky.notification.list_notifications().notifications

        for note in notifications:
            if note.reason == "mention" and note.uri not in processed_notifications:
                reply_with_pdf(note)
                processed_notifications.add(note.uri)

        time.sleep(100)

if __name__ == "__main__":
    main()