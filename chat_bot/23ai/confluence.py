from atlassian import Confluence
import re
from bs4 import BeautifulSoup
confluence = Confluence(
    url='https://confluence.oraclecorp.com/confluence',
    token='***'
)

page_id = '5203270386'
# page_id = '3686238315'
page_content = confluence.get_page_by_id(page_id, expand='body.storage')['body']['storage']['value']

# print(page_content)
# breakpoint()
soup = BeautifulSoup(page_content, 'lxml')

# tables = soup.find_all('table')
# data = []
#
# for table in tables:
#     table_data = []
#     rows = table.find_all('tr')
#     for row in rows:
#         cols = row.find_all(['td', 'th'])
#         cols = [ele.text.strip() for ele in cols]
#         table_data.append(cols)
#         print(cols)
#     data.append(table_data)
# breakpoint()


def get_conf_data():
    structured_data = []
    row_no = 1
    # Extract paragraphs and other text elements outside of tables
    # paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li'])
    paragraphs = soup.select(
        'p:not(table p), h1:not(table h1), h2:not(table h2), h3:not(table h3), '
        'h4:not(table h4), h5:not(table h5), h6:not(table h6), '
        'ul:not(table ul), ol:not(table ol), li:not(table li)')
    for para in paragraphs:
        # text = re.sub(r'\s+', ' ', para.get_text()).strip()
        text = para.get_text().strip()
        if text:  # Ensure non-empty text
            structured_data.append({
                'row_no': row_no,
                'title': text,
                'content': text
            })
            row_no += 1

    # Extract tables
    tables = soup.find_all('table')

    for table in tables:
        # table_data = []
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['td', 'th'])
            title = cols[1].text.strip() if len(cols) > 1 else cols[0].text.strip()
            # title = ' | '.join(ele1.text.strip() for ele1 in cols[0:1])
            row_text = ' | '.join(ele.text.strip() for ele in cols)
            # cols = [re.sub(r'\s+', ' ', ele.text).strip() for ele in cols]
            # cols = [ele.text.strip() for ele in cols]
            # structured_data.append(cols)
            structured_data.append({
                'row_no': row_no,
                'title': title,
                'content': row_text
            })
            row_no+=1
            # print(row_text)
        # data.append(table_data)
    print(structured_data)

    # print(len(structured_data))
    return structured_data

get_conf_data()

for each in get_conf_data():
    print('#'*15)
    print(each)