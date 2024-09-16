import oci
import oracledb
import streamlit as st

cs = """(description = (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)
(host=***))(connect_data=
(service_name=***))
(security=(ssl_server_dn_match=yes)))"""

OCI_CRED = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url":
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v2.0"
    }

connection = oracledb.connect(
     user="admin",
     password='Dev0ps@12345',
     dsn=cs)

cursor = connection.cursor()
cursor.callproc("DBMS_OUTPUT.ENABLE")


def get_dbms_output(_cursor, numlines=10):
    lines = []
    for _ in range(numlines):
        line_var = _cursor.var(str)
        status_var = _cursor.var(int)
        _cursor.callproc("DBMS_OUTPUT.GET_LINE", (line_var, status_var))
        if status_var.getvalue() != 0:
            break
        lines.append(line_var.getvalue())
    return lines


VECTOR_DISTANCE = """
DECLARE
    user_input      VARCHAR2(4000) := '{0}';
    query_vector    VECTOR;
BEGIN
    query_vector := dbms_vector.utl_to_embedding(user_input, json('{{
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-v3.0"
    }}'));

    FOR r IN (
        SELECT CRT.TITLE, CRT.CHUNK_ID, VECTOR_DISTANCE(CRT.VECTOR, query_vector, COSINE) AS distance
        FROM CONFLUENCE_ROW_TITLE CRT
        JOIN CONFLUENCE_ROW_CONTENT CRC
        ON CRT.CHUNK_ID = CRC.ROW_ID
        ORDER BY distance ASC
        FETCH FIRST {1} ROWS ONLY
    ) LOOP
        DBMS_OUTPUT.PUT_LINE('Sentence: ' || r.title || ', Chunk_id: ' || r.CHUNK_ID || ', Similarity (COSINE): ' || r.distance);
        DBMS_OUTPUT.PUT_LINE(r.CHUNK_ID);
    END LOOP;
END;
"""

QUERY_TO_GET_DATA_BY_ROW_ID = """
DECLARE
    v_concatenated_content CLOB := '';
BEGIN
    FOR r IN (
        SELECT CRT.TITLE, CRC.CONTENT, CRT.CHUNK_ID
        FROM CONFLUENCE_ROW_TITLE CRT
        JOIN CONFLUENCE_ROW_CONTENT CRC
        ON CRT.CHUNK_ID = CRC.ROW_ID
        WHERE CRC.ROW_ID = {0}
        order by CRC.ID
    ) LOOP
        -- Concatenate each content with a space
        v_concatenated_content := v_concatenated_content || r.CONTENT || ' ';
    END LOOP;

    -- Display the concatenated result
    DBMS_OUTPUT.PUT_LINE(v_concatenated_content);
END;
"""

config = oci.config.from_file()
ai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)


def query_vector_db(_query, top_k=1):
    return cursor.execute(VECTOR_DISTANCE.format(_query, top_k))


def generate_response_with_llm(_query, steps):
    context = "".join(steps)
    instructions = """
    You are a chat bot. You need to answer questions based on the text provided above.
    If the text doesn't contain the answer, don't go outside the data you have.
    Be soft and answer in a better way because you're answering to my customers.
    """
    input_text = f"Context: {context}\n\nQuestion: {_query}\n\nAnswer:  " \
                 f"(In bullet points, user friendly output; skip execution log from context; " \
                 f"Additional Instruction {instructions})"
    compartment_id = \
        "ocid1.compartment.oc1..aaaaaaaakggm6zsow2fefyjbtvftjdd7bxkgmvazunepkpi34o6hpzzequca"
    _config = oci.config.from_file('~/.oci/config', "DEFAULT")

    # Service endpoint
    endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=_config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240))
    generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
    llm_inference_request = oci.generative_ai_inference.models.CohereLlmInferenceRequest()
    llm_inference_request.prompt = input_text
    llm_inference_request.max_tokens = 800
    llm_inference_request.temperature = 1
    llm_inference_request.frequency_penalty = 0
    llm_inference_request.top_p = 0.75

    generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode \
        (model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyafhwal37hxwylnpbcncidimbwteff4xha77n5xz4m7p6a")
    generate_text_detail.inference_request = llm_inference_request
    generate_text_detail.compartment_id = compartment_id
    generate_text_response = generative_ai_inference_client.generate_text(generate_text_detail)
    # Print result
    return generate_text_response.data.inference_response.generated_texts[0].text


def get_id_list(_result):
    _results = cursor.execute(QUERY_TO_GET_DATA_BY_ROW_ID.format(_result))
    return _results


# if 'selected_task' not in st.session_state:
#     st.session_state.selected_task = None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Streamlit Interface
st.title("Chat Bot for internal runbooks using RAG")
# Example query
query = st.text_input("Enter your query:")
if query:
    with st.status("Getting your data...", expanded=True) as status1:
        st.write("Searching for data...")

        chat_context = "\n".join(
            [f"User: {chat['query']}\nAssistant: {chat['response']}" for chat in
             st.session_state.chat_history])
        full_prompt = f"{chat_context}\nUser: {query}\nAssistant:"
        results = query_vector_db(query)
        output = get_dbms_output(cursor)

        top_k_results = get_id_list(output[-1])
        output1 = get_dbms_output(cursor)

        response_with_llm = generate_response_with_llm(full_prompt, output1[0])
        st.session_state.chat_history.append({"query": query, "response": response_with_llm})
        st.write(response_with_llm)
        status1.update(label="Download complete!", state="complete", expanded=True)
    st.button("Rerun")


st.write("## Chat History")
for i, chat in enumerate(st.session_state.chat_history, 1):
    st.write(f"**Question {i}:** {chat['query']}")
    st.write(f"**Response {i}:** {chat['response']}")


# Reset chat button
if st.button("Reset Chat"):
    st.session_state.chat_history = []

# query = 'Get me steps for "Dropping the Initial provisioned Pluggable Database (PDB) at OCI"'
#
# results = query_vector_db(query)
# output = get_dbms_output(cursor)
#
# breakpoint()
#
# # Fetch the DBMS_OUTPUT
# top_k_results = get_id_list(output[-1])
# output1 = get_dbms_output(cursor)
# # top_k_results = [get_id_list(result) for result in results['matches']][0]
# # # summaries = [summarize_text(result)+'******\n' for result in top_k_results]
# response_with_llm = generate_response_with_llm(query, output1[0])
# # print("**************************Generate Texts Result**************************")
# print(response_with_llm)
