# syntax=docker/dockerfile:experimental
FROM quay.io/unstructured-io/base-images:rocky8.7-3 as base

ARG PIP_VERSION

# Set up environment
ENV HOME /home/
WORKDIR ${HOME}
RUN mkdir ${HOME}/.ssh && chmod go-rwx ${HOME}/.ssh \
  &&  ssh-keyscan -t rsa github.com >> /home/.ssh/known_hosts
ENV PYTHONPATH="${PYTHONPATH}:${HOME}"
ENV PATH="/home/usr/.local/bin:${PATH}"

FROM base as deps
# Copy and install Unstructured
COPY requirements requirements

RUN python3.8 -m pip install pip==${PIP_VERSION} && \
  dnf -y groupinstall "Development Tools" && \
  pip install --no-cache -r requirements/base.txt && \
  pip install --no-cache -r requirements/test.txt && \
  pip install --no-cache -r requirements/huggingface.txt && \
  pip install --no-cache -r requirements/dev.txt && \
  pip install --no-cache -r requirements/ingest-azure.txt && \
  pip install --no-cache -r requirements/ingest-github.txt && \
  pip install --no-cache -r requirements/ingest-gitlab.txt && \
  pip install --no-cache -r requirements/ingest-google-drive.txt && \
  pip install --no-cache -r requirements/ingest-reddit.txt && \
  pip install --no-cache -r requirements/ingest-s3.txt && \
  pip install --no-cache -r requirements/ingest-slack.txt && \
  pip install --no-cache -r requirements/ingest-wikipedia.txt && \
  pip install --no-cache -r requirements/local-inference.txt && \
  pip install --no-cache Flask==2.2.3 && \
  pip install --no-cache langchain==0.0.245 && \
  pip install --no-cache langchain_experimental==0.0.5 && \
  pip install --no-cache pdfminer.six && \
  pip install --no-cache pinecone-client && \
  pip install --no-cache requests && \
  pip install --no-cache openai && \
  pip install --no-cache tiktoken && \
  pip install --no-cache azure-storage-blob azure-identity && \
  pip install --no-cache azure-messaging-webpubsubservice && \
  pip install --no-cache deepl && \
  pip install --no-cache azure-search-documents==11.4.0b6 && \
  dnf -y groupremove "Development Tools" && \
  dnf clean all

RUN python3.8 -c "import nltk; nltk.download('punkt')" && \
  python3.8 -c "import nltk; nltk.download('averaged_perceptron_tagger')"

FROM deps as code  

COPY example-docs example-docs
COPY unstructured unstructured

RUN python3.8 -c "from unstructured.ingest.doc_processor.generalized import initialize; initialize()"

COPY myapp myapp
COPY run.py run.py

# Make port 80 available to the world outside this container
EXPOSE 8080
EXPOSE 443

CMD ["python3.8", "run.py"]
# CMD ["/bin/bash"]


