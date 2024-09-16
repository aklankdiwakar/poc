import oracledb
# from transformers import AutoTokenizer, AutoModel
from confluence import get_conf_data

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')


cs="""(description = (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=***))(connect_data=(service_name=***))(security=(ssl_server_dn_match=yes)))"""
OCI_CRED = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v2.0"
    }

connection = oracledb.connect(
     user="admin",
     password='***',
     dsn=cs)

cursor = connection.cursor()
cursor.callproc("DBMS_OUTPUT.ENABLE")
structured_text = get_conf_data()


# Store embeddings with metadata
def store_embeddings(_segments, _row_no, page_id=1):
    try:
        for i, segment in enumerate(_segments):
            insert_command = f'''
            DECLARE
                input_list  SYS.ODCIVARCHAR2LIST;
                params      CLOB;
                v           VECTOR;
            BEGIN
                input_list := SYS.ODCIVARCHAR2LIST('{_row_no}{i}', '{segment.replace("'", "''")}');
                params := '{{
                "provider": "ocigenai",
                "credential_name": "OCI_CRED",
                "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
                "model": "cohere.embed-english-v3.0"
                }}';
                BEGIN
                    v := dbms_vector.utl_to_embedding(input_list(2), json(params));
                    INSERT INTO CONFLUENCE_ROW_TITLE (PAGE_ID, CHUNK_ID, TITLE, VECTOR)
                    VALUES ({page_id}, input_list(1), input_list(2), v);
                    dbms_output.put_line('Embedding for "' || input_list(1) || '" inserted successfully.');
                EXCEPTION
                    WHEN OTHERS THEN
                        DBMS_OUTPUT.PUT_LINE('Error processing sentence: "' || input_list(1) || '"');
                        DBMS_OUTPUT.PUT_LINE(SQLERRM);
                        DBMS_OUTPUT.PUT_LINE(SQLCODE);
                END;
                COMMIT;
            END;
            '''
            # print(insert_command)
            cursor.execute(insert_command)
    except Exception as err:
        print(err)


def store_content(_segments, _page_num):
    # breakpoint()
    try:
        for i, segment in enumerate(_segments):
            insert_commend = f'''INSERT INTO CONFLUENCE_ROW_CONTENT (ROW_ID, TOTAL_CHUNK, CONTENT) VALUES ({_page_num}{0}, {len(_segments)}, '{segment.replace("'", "''")}')'''
            print(insert_commend)
            cursor.execute(insert_commend)
        cursor.connection.commit()
    except Exception as error:
        print(f'error : {error}')
        # breakpoint()

# def save_text_to_file(text, _output_path):
#     with open(_output_path, 'w') as file:
#         file.write(text)


# def embed_text(text):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
#     return embeddings.flatten().tolist()

# structured_text = extract_text_from_pdf(pdf_path)
# for data in structured_text:
#     # Split text into chunks
#     try:
#         title_segments = [data['title'][i:i + 512] for i in range(0, len(data['title']), 512)]
#         store_embeddings(title_segments, data['row_no'])
#     except Exception as error:
#         print(f'error: {error}')

for data in structured_text:
    # Split text into chunks
    try:
        content_segments = [data['content'][i:i + 512] for i in range(0, len(data['content']), 512)]
        store_content(content_segments, data['row_no'])
    except Exception as error:
        print(f'error: {error}')
        # breakpoint()
    # embeddings = [embed_text(segment) for segment in segments]
    # print(data['page'])


# Fetch and print the DBMS_OUTPUT results
status_var = cursor.var(int)
while True:
    line_var = cursor.var(str)
    cursor.callproc("DBMS_OUTPUT.GET_LINE", (line_var, status_var))
    if status_var.getvalue() != 0:
        break
    print(line_var.getvalue())

# Close the cursor and connection
cursor.close()
connection.close()
# save_text_to_file(structured_text, output_path)
# print(f"Extracted text saved to {output_path}")
