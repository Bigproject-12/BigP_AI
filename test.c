#include <stdio.h>
#include <string.h>

char *password = "1234";

char *get_password() {
    return password;
}

char *hash_it(char *p) {
    static char buf[32];
    sprintf(buf, "hashed_%s", p);
    return buf;
}

int main() {
    printf("Password: %s\n", password);
    printf("%s\n", hash_it(password));
    printf("%s\n", get_password());
    return 0;
}