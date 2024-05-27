import re

def get_surrounding_words(text, target, num_words=5):
    # Find all words in the text
    words = re.findall(r'\w+', text)
    
    # Find the position of the target word
    target_words = re.findall(r'\w+', target)
    target_len = len(target_words)
    
    for i in range(len(words)):
        if words[i:i + target_len] == target_words:
            start = max(i - num_words, 0)
            end = min(i + target_len + num_words, len(words))
            return ' '.join(words[start:end])
    
    return None

# Example usage
large_text = "The quick brown fox jumps over the lazy dog. The rain in Spain stays mainly in the plain. To be or not to be, that is the question."
target_string = "jumps over"
num_surrounding_words = 7

result = get_surrounding_words(large_text, target_string, num_surrounding_words)
print(result)
