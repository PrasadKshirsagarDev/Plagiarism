from urllib.parse import unquote

url = "https://hi.wikipedia.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A4%BF%E0%A4%AA%E0%A5%80%E0%A4%A1%E0%A4%BF%E0%A4%AF%E0%A4%BE:%E0%A4%95%E0%A5%8D%E0%A4%AF%E0%A4%BE_%E0%A4%86%E0%A4%AA_%E0%A4%9C%E0%A4%BE%E0%A4%A8%E0%A4%A4%E0%A5%87_%E0%A4%B9%E0%A5%88_(%E0%A4%AA%E0%A5%82%E0%A4%B0%E0%A5%8D%E0%A4%B5-%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A4%A6%E0%A4%B0%E0%A5%8D%E0%A4%B6%E0%A4%BF%E0%A4%A4)"
decoded_url = unquote(url)

print(decoded_url)
