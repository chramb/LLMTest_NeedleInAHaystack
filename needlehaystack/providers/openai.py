import os
from operator import itemgetter
from typing import TYPE_CHECKING, Optional

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
import tiktoken

from .model import ModelProvider

if TYPE_CHECKING:
    import tiktoken
    import tokenizers

class OpenAI(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 tokenizer: str = "tiktoken:auto",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the OpenAI model provider with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs

        found_tokenizer = self.get_encoder(model_name, tokenizer)
        if found_tokenizer is None:
            msg = f"Could not find correct tokenizer based on string: {tokenizer}"
            raise ValueError(msg)

        self.tokenizer = found_tokenizer
    
    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response = await self.model.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **self.model_kwargs
            )
        return response.choices[0].message.content
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        encoding: list[int] | "tokenizers.Encoding" = self.tokenizer.encode(text)
        if hasattr(encoding, "ids"):
            return encoding.ids
        else:
            return encoding
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatOpenAI(temperature=0, model=self.model_name)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain
    
    def get_encoder(self, model_name: str, tokenizer_arg: str) -> "tiktoken.Encoding | tokenizers.Tokenizer | None":
        parts = tokenizer_arg.split(":",1)
        module = parts[0]

        if module == "tiktoken":
            import tiktoken
            try:
                if len(parts) < 2 or parts[1] == "auto":
                    return tiktoken.encoding_for_model(model_name)
                else:
                    return tiktoken.get_encoding(parts[1])
            except (ValueError, KeyError) as e:
                msg = f"Encoding not Found, available encodings: {tiktoken.list_encoding_names()}"
                raise ValueError(msg) from e

        if len(parts) < 2:
            msg = "Missing path part of tokenizer argument."
            raise ValueError(msg)

        if module in ("tokenizers", "hf", "huggingface"):
            from tokenizers import Tokenizer
            return Tokenizer.from_pretrained(parts[1])

        if module == "file":
            from tokenizers import Tokenizer
            return Tokenizer.from_file(parts)

        msg = f"tokenizer string '{tokenizer_arg}' could not be resolved to any tokenizer."
        raise ValueError(msg)