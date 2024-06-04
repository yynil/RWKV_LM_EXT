from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common import NoSuchElementException, ElementNotInteractableException

firefox_path = '/home/yueyulin/firefox/geckodriver'
service = Service(firefox_path)
options = webdriver.FirefoxOptions()
options.binary_location = '/home/yueyulin/firefox/firefox'
driver = webdriver.Firefox(service=service,options=options)

driver.get("https://www.baidu.com/")

title = driver.title


text_box = driver.find_element(by=By.ID, value="kw")
submit_button = driver.find_element(by=By.ID, value="su")
def get_texts(driver):
    paragraps = driver.find_elements(By.CSS_SELECTOR, "div[data-tag='paragraph']")
    p_results = []
    for p in paragraps:
        texts_element = p.find_elements(By.CSS_SELECTOR, ".text_jaYku[data-text='true']")
        texts = ''.join([text.text for text in texts_element])
        p_results.append(texts)
    return p_results

text_box.send_keys("黄循财生日什么时候？")
submit_button.click()
wait = WebDriverWait(driver, 10)
#wait until the next page is displayed
errors = [NoSuchElementException, ElementNotInteractableException]
wait = WebDriverWait(driver, timeout=5, poll_frequency=.2, ignored_exceptions=errors)
wait.until(lambda driver : driver.find_element(by=By.CLASS_NAME, value="c-container")!= None)
print('loaded')
containers = driver.find_elements(by=By.CLASS_NAME, value="c-container")
#filter the containers with CLASS_NAME also has result
for container in containers:
    #find the href and text of the container
    href = container.find_element(by=By.TAG_NAME, value="a")
    print(href.get_attribute("href"))
    print(href.text)
    print(container.get_attribute("class"))
    print(href.get_attribute("class"))
    print('-----------------')
    if href.text.endswith('_百度百科'):
        driver.get(href.get_attribute("href"))
        wait = WebDriverWait(driver, timeout=5, poll_frequency=.2, ignored_exceptions=errors)
        wait.until(lambda driver : driver.find_element(by=By.CLASS_NAME, value="text_jaYku")!= None)
        print('baike loaded')
        texts = get_texts(driver)
        print(texts)
        break

driver.quit()