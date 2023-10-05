from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
print("loaded all packages")

def llama2_main_function(file):

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    print("Printing Device...")
    print(device)

    print("loading model....")
    model_id ="meta-llama/Llama-2-7b-chat-hf"
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit = True,
    bnb_4bit_quant_tyoe = 'nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
    )


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth,
        offload_folder="save_folder"
    )


    # Load model directly
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)



    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True, 
        task='text-generation',
        temperature=0.1,  
        max_new_tokens=4096,  
        repetition_penalty=1.1 
    )

    print("loaded model")


    llm = HuggingFacePipeline(pipeline=generate_text)


   
    sm_loader = UnstructuredFileLoader(file)
    sm_doc = sm_loader.load()

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)


    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 6000,
        chunk_overlap = 400
    )

    sam_docs = text_splitter.split_documents(sm_doc)

    result = chain.run(sam_docs)

    return result

