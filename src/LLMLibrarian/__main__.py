from LLMLibrarian.librarian import Librarian

def main():
    librarian = Librarian(base_url="http://localhost:8080/v1", api_key="sk-no-key-required", num_chunks=20)
    folder = input("Enter the path to the folder you would like to process: ")
    librarian.process_files_in_directory(folder)

    while True:
        query = input("Enter a query to search for in the library: ")
        answer = librarian.query(query)
        print(answer)

if __name__ == "__main__":
    main()
