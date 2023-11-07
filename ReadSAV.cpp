#include <readstat.h>

int main() {
    const char* filename = "exp1.sav";

    readstat_error_t error = READSTAT_OK;
    readstat_parser_t* parser = readstat_parser_init();

    // Set up your callbacks and data structures for handling the data.

    if (readstat_parse_file(parser, filename, NULL) != READSTAT_OK) {
        error = readstat_error(parser);
        fprintf(stderr, "Error processing file: %s\n", readstat_error_message(error));
    } else {
        // Process the data as needed.
    }

    readstat_parser_free(parser);

    return 0;
}
