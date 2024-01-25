
<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h3 align="center"> Tax Tribunal Case Analysis (TTCA)</h3>
  <p align="center">
    Legal Data Product for First-tier Tribunal decisions.
    <br />
    <a href="https://github.com/starrywheat/gov_tax_data"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://app.hex.tech/a13d28df-9014-440d-ac6d-49fa575ec88b/app/408021b6-808b-47f0-8111-c14cf5cf3f48/latest">View Demo</a>
    ·
    <a href="https://github.com/starrywheat/gov_tax_data/issues">Report Bug</a>
    ·
    <a href="https://github.com/starrywheat/gov_tax_data/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#extracting-legal-knowledge-with-openai">Extracting Legal Knowledge with OpenAI</a>
      <ul>
        <li><a href="#technical-setup">Technical setup</a></li>
        <li><a href="#run-the-streamlit-app-locally"> Run the streamlit app locally</a></li>
      </ul>
    </li>
    <li><a href="#experimentation-with-information-retrieval-and-prompting">Experimentation with Information retrieval and prompting</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
This is a research project aim analyse roughly 7,500 Tax Tribunal decisions to answer through data:​

- Which decisions concern tax avoidance ?

- What is the outcome of each decision (i.e. who won)?

- Other insights and trends (e.g. HMRC win rate over the years).​


This project showcases the use of ChatGPT to respond to the questions above with different approaches to extract relevant information from the case documents.

The output of this work consist of two products:
- A Streamlit🎈[app](https://streamlit.io/) to query on the public Tax Tribunal documents, powered by ChatGPT.
- An interactive [dashboard](https://app.hex.tech/a13d28df-9014-440d-ac6d-49fa575ec88b/app/408021b6-808b-47f0-8111-c14cf5cf3f48/latest) that summarises the findings of the data analysis

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built with

* [OpenAI](https://openai.com/blog/introducing-gpts)
* [Langsmith](https://www.langchain.com/langsmith)
* [Streamlit🎈](https://streamlit.io/)
* [Pinecone🗂️](https://www.pinecone.io/)
* [Hex](https://hex.tech/)
* [Azure Cloud](https://azure.microsoft.com/en-us/get-started/azure-portal)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Methodology -->
## Extracting Legal Knowledge with OpenAI

Tribunal case decisions report are complicated and do not have standard formats, meaning pattern/keyword search is not feasible to extract answers to the above legal questions. That's where LLM,  come into play as it is good at doing generative question-answering.
I approach this by building a **Retrieval Augmented Generation (RAG)** pipeline to query the set of tribunal case decisions. I also build a **streamlit🎈** interface to help doing experiments on prompt development.

The interface gives you two ways to interact with the cases
1. A simple question-answer approach, in which you supply with a user query, and the pipeline that connects to **OpenAI API** would response with an answer. This is a great way to experiment with your prompts for prompt developments
2. A conversational agent, which is a chatbot style interface for the user to have multiple conversations. The agent contains memories of the conversations, therefore users can follow up on previous LLM responses.

<!-- TECHNICAL SETUP -->
### Technical Setup
The app allows you to query the already indexed Tribunal cases that are stored in **pinecone🗂️ vectorDB**, on a case by case basis. Here are the steps of how I setup my environment to get the **streamlit🎈** app up and running:

1. Set up your python environment, e.g. with anaconda
```
conda create -n ttca python=3.10
conda activate ttca
pip install -r requirements.txt
```
2. Sign up for an OpenAI API [account](https://auth0.openai.com/u/signup/identifier?state=hKFo2SBWaWJrYmJZNEtnWDcxcmtkSjh3Mmd2VkktX2kydEphN6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDZMXzJyaDhzcHAybkM3cFlOVXQyWnBoSXRMZ25yQllIo2NpZNkgRFJpdnNubTJNdTQyVDNLT3BxZHR3QjNOWXZpSFl6d0Q) and get API key


3. Sign up for a free account on [Pinecone](https://www.pinecone.io/). A free account would allow creation of a single index.

4. (Optional) If you want to trace your LLM calls, sign up for an account on [langsmith](https://www.langchain.com/langsmith) and get the API key

5. (Optional) If you want to use langchain Azure Blob loader, sign up for a free Azure portal and create a blob storage account. Get the connection strings of the storage account.

6. Save all the relevant credentials in .streamlit/secrets.toml (See an example in the directory)

### Run the streamlit app locally
Run the streamlit app locally
```
streamlit run ttca/Main.py
```
A new window will pop up in the browser to show you the app. Otherwise, go to `http://localhost:8501` in your browser to access the app.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Experimentation with Information retrieval and prompting

RAG may not be the most efficient and accurate way to retrieve knowledge from documents. Different questions require different levels of information retrievals. I implement various classes to tackle various types of user queries. Check `ttca/utils_llm.py`

### Specific Chunk
The information needed to respond to a user query is always located in the same section of a document.
>💡Example use case: What is the decision of the court case?

### Map reduce
The information needed to answer a user query is often distributed throughout a document.
>💡Example use case: Give the top 10 main topics discussed in court case.

### RAG
The necessary information to respond to the user's query might be located in a few paragraphs whose locations are not known.
>💡Example use case: Determine whether the tax avoidance is involved in this court case.




<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
