import re

URGENCY_WORDS = ["urgent", "immediate", "action required", "verify", "asap", "attention"]
CREDENTIAL_WORDS = ["password", "login", "credentials", "account"]
SUSPICIOUS_DOMAINS = [".ru", ".tk", ".cn", ".info"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co"]

''' 
helper functions needed:
- extracting host
- checking if IP host is valid
'''

def link_rules(email_text):
    score = 0

    # IP-based URls
    ip_pattern = r"https?://(?:\d{1,3}\.){3}\d{1,3}(?:[/:]\S*)?"
    if re.search(ip_pattern, email_text):
        score += 3

    # url detection and count
    url_pattern = r"(https?://\S+|www\.\S+)"
    urls = re.findall(url_pattern, email_text.lower())

    if len(urls) >= 3:
        score += 3
    elif len(urls) >= 1:
        score += 2

    # suspicious domains
    for u in urls:
        host = re.search(r"https?://([^/]+)", u)
        if host and host.group(1).split(":")[0] in SHORTENERS:
            score += 2

    return score

def language_rules(email_text):
    score = 0 
    
    # detecting urgency and credential requests
    if any(word in email_text.lower() for word in URGENCY_WORDS):
        score += 1

    if any(word in email_text.lower() for word in CREDENTIAL_WORDS):
        score += 1
    
    return score

def formatting_rules(email_text):
    score = 0 

    # excessive use of punctuation and uppercase letters
    if email_text.count("!") > 3:
        score += 1

    caps_words = [w for w in re.findall(r"[A-Za-z]+", email_text) if w.isupper() and len(w) > 3]
    if len(caps_words) >= 3:
        score += 1

    return score

# combining link, language and formatting rules for overall risk score
def final_risk_score(email_text):
    risk_score = 0
    risk_score += link_rules(email_text)
    risk_score += language_rules(email_text)
    risk_score += formatting_rules(email_text)

    return risk_score
