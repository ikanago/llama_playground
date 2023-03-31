import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor


def main():
    pwd = __file__
    pwd_dir = pwd.rsplit("/", 1)[0]
    index_dir = os.path.join(pwd_dir, "index")
    index_file = os.path.join(index_dir, "index.json")

    if not os.path.exists(index_file):
        os.makedirs(index_dir, exist_ok=True)

        documents = SimpleDirectoryReader(os.path.join(pwd_dir, "data")).load_data()
        service_context = ServiceContext.from_defaults(
            llm_predictor=ChatGPTLLMPredictor()
        )
        index = GPTSimpleVectorIndex.from_documents(
            documents, service_context=service_context
        )
        index.save_to_disk(index_file)
    else:
        service_context = ServiceContext.from_defaults(
            llm_predictor=ChatGPTLLMPredictor()
        )
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file,
            service_context=service_context
        )
    while True:
        inp = input("What's your question?\n> ")
        print(index.query(inp))
        print()


if __name__ == "__main__":
    main()
