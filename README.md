# Inkwell AI - bluesky-ebook-generator-ai-bot
A bluesky AI bot which creates automated threads, can generate ebooks and interact with users .

Bot Username :
@curosepian.bsky.social - Inkwell AI
link : https://bsky.app/profile/curosepian.bsky.social

features:
1. Automated Thread generator :

Bot created automated thread after every 30 minutes on diffrent thought provoking topics

2. Ebook generator :

Bot can generate ebook on given topic on mention. It consists of three agents as follows

Researcher Agent: Gathers information and generates the structure (chapters) relevant to the specified topic.

Writer Agent: Writes the content for each chapter based on the research.

Editor Agent: Corrects grammatical mistakes and enhances the content for clarity and readability.


Also we can configure the ebook generation using following constraints

- format - format of generated ebook. Currently, we are supporting 3 file format

text - for ebook as .txt file
doc - for ebook as .doc file
pdf - for ebook as .pdf file

- chapters - total number of chapters in ebook (max=20 chapters)

- writing style - Writing style of ebook. Following writing styles are supported

"Narrative", "Expository", "Professional", "Classy", 
"Persuasive", "Descriptive", "Conversational", "Satirical", 
"Technical", "Inspirational", "Analytical", "Epic", "Normal"

3. Reply on Mention :

Bot can reply normally or suggest books on mention.



Deployed here:

hugging face link

https://huggingface.co/spaces/suryanshp1/bsky-bot


Technology Used:
- Python
- langchain
- GROQ
- Cloudinry
- Atproto


Examples:

1. Ebook generation example text

```
@curosepian.bsky.social
generate a pdf ebook on Personal finance and investing with 12 chapters in professional writing style
```

2. Ebook suggestion

```
@curosepian.bsky.social
give me some trending topics to generate ebooks
```

3. Normal Interaction

```
@curosepian.bsky.social
Who is the father of AI ?
```