import os
import sys
import csv
import torch
import threading
import matplotlib.pyplot as plt
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QTabWidget, QFormLayout, QDateEdit, QProgressBar
)
from PyQt6.QtCore import QDate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 필요한 라이브러리 설치
try:
    import sentencepiece
except ImportError:
    os.system("pip install sentencepiece")

try:
    import google.protobuf
except ImportError:
    os.system("pip install protobuf")

# OpenMP 충돌 해결
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# KoBERT 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3, trust_remote_code=True)

# 난수 시드 설정
torch.manual_seed(42)


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_page_tab(), "Page 기반 크롤링")
        self.tabs.addTab(self.create_date_tab(), "Date 기반 크롤링")

        # 실행 버튼
        self.run_button = QPushButton("실행")
        self.run_button.clicked.connect(self.run_analysis)

        # 진행 상황 표시를 위한 ProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # 결과 라벨
        self.result_label = QLabel("결과가 여기에 표시됩니다.")

        layout.addWidget(self.tabs)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle("감성 분석 프로그램")
        self.setGeometry(100, 100, 400, 400)

    def create_page_tab(self):
        page_tab = QWidget()
        layout = QFormLayout()

        self.board_select = QComboBox()
        self.board_select.addItem("CPU", "https://quasarzone.com/bbs/qf_cmr")
        self.board_select.addItem("그래픽카드", "https://quasarzone.com/bbs/qf_vga")
        self.board_select.addItem("키보드/마우스", "http://quasarzone.com/bbs/qf_input")
        layout.addRow("게시판 선택:", self.board_select)

        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("키워드를 입력하세요")
        layout.addRow("키워드:", self.keyword_input)

        self.page_count_input = QSpinBox()
        self.page_count_input.setRange(1, 100)
        self.page_count_input.setValue(3)
        layout.addRow("크롤링할 페이지 수:", self.page_count_input)

        page_tab.setLayout(layout)
        return page_tab

    def create_date_tab(self):
        date_tab = QWidget()
        layout = QFormLayout()

        self.board_select_date = QComboBox()
        self.board_select_date.addItem("CPU", "https://quasarzone.com/bbs/qf_cmr")
        self.board_select_date.addItem("그래픽카드", "https://quasarzone.com/bbs/qf_vga")
        self.board_select_date.addItem("키보드/마우스", "http://quasarzone.com/bbs/qf_input")
        layout.addRow("게시판 선택:", self.board_select_date)

        self.keyword_input_date = QLineEdit()
        self.keyword_input_date.setPlaceholderText("키워드를 입력하세요")
        layout.addRow("키워드:", self.keyword_input_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        layout.addRow("종료 날짜:", self.end_date)

        date_tab.setLayout(layout)
        return date_tab

    def run_analysis(self):
        if self.tabs.currentIndex() == 0:
            url = self.board_select.currentData()
            keyword = self.keyword_input.text()
            page_count = self.page_count_input.value()
            # 크롤링 작업을 별도의 스레드에서 실행
            threading.Thread(target=self.crawl_by_page, args=(url, keyword, page_count), daemon=True).start()
        else:
            url = self.board_select_date.currentData()
            keyword = self.keyword_input_date.text()
            end_date = self.end_date.date().toString("MM-dd")
            # 크롤링 작업을 별도의 스레드에서 실행
            threading.Thread(target=self.crawl_by_date, args=(url, keyword, end_date), daemon=True).start()

    def crawl_by_page(self, url, keyword, page_count):
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)



        visited_titles = set()

        with open("sentiment.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Title"])

            # 총 게시물 수 계산
            total_posts = 0
            for page_num in range(1, page_count + 1):
                driver.get(f"{url}?page={page_num}")
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "tit")))
                    posts = driver.find_elements(By.CLASS_NAME, "tit")
                    total_posts += len(posts)
                except TimeoutException:
                    break

            # 크롤링 진행
            current_post = 0
            for page_num in range(1, page_count + 1):
                driver.get(f"{url}?page={page_num}")
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "tit")))
                    posts = driver.find_elements(By.CLASS_NAME, "tit")

                    for post in posts:
                        try:
                            a_tag = post.find_element(By.CSS_SELECTOR, "a")
                            title_text = a_tag.text
                            post_url = a_tag.get_attribute("href")

                            if keyword.lower() in title_text.lower() and title_text not in visited_titles:
                                visited_titles.add(title_text)
                                driver.execute_script(f"window.open('{post_url}', '_blank');")
                                driver.switch_to.window(driver.window_handles[1])
                                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "note-editor")))
                                body = driver.find_element(By.CLASS_NAME, "note-editor").text
                                body = body.replace('\n', ' ').replace('\r', '').replace(',', '')
                                writer.writerow([title_text])
                                driver.close()
                                driver.switch_to.window(driver.window_handles[0])
                        except NoSuchElementException:
                            continue

                        current_post += 1
                        progress = int((current_post / total_posts) * 100)
                        self.progress_bar.setValue(progress)

                except TimeoutException:
                    break

            # 크롤링 완료 후 progress_bar를 100%로 설정
            self.progress_bar.setValue(100)

        driver.quit()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec())