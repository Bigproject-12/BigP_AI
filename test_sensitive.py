password = "1234"

def get_password():
    return password

def hash_it(p):
    return str(hash(p))

print("Password:", password)
print(hash_it(password))
print(get_password())
