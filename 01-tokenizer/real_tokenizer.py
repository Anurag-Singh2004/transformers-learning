import tiktoken

# GPT-4's actual tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# we will see encode, decode

#Basic encoding
print(enc.encode("hello world"))
print(enc.encode(" hello world"))

#The strawberry problem
print(enc.encode("strawberry"))

#see actual token strings
tokens = enc.encode("tokenization is interesting")
for token_id in tokens:
    print(f"{token_id} → '{enc.decode([token_id])}'") #decoding string from integers