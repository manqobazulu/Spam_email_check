{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO60kTlYR4UDT2+8IS/JiH0",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/manqobazulu/Spam_email_check/blob/main/215273024_ZuluTM_SpamOrNot_Email.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "EXj4Zg3pRNYj"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score\n",
        "import pickle"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 1: Dataset Preparation\n",
        "# Assuming you have a CSV file containing the dataset with columns 'text' and 'label' (0 for non-spam, 1 for spam)\n",
        "# Adjust the file path accordingly.\n",
        "# data = pd.read_csv('path/to/dataset.csv')\n",
        "data = pd.read_csv(\"/content/spam_ham_dataset.csv\")"
      ],
      "metadata": {
        "id": "s8LmWggwRoYv"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "MhGa2T6dTxs8",
        "outputId": "fe06c37f-ac56-4e3f-d308-5e0baf638f96"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Unnamed: 0 label                                               text  \\\n",
              "0         605   ham  Subject: enron methanol ; meter # : 988291\\r\\n...   \n",
              "1        2349   ham  Subject: hpl nom for january 9 , 2001\\r\\n( see...   \n",
              "2        3624   ham  Subject: neon retreat\\r\\nho ho ho , we ' re ar...   \n",
              "3        4685  spam  Subject: photoshop , windows , office . cheap ...   \n",
              "4        2030   ham  Subject: re : indian springs\\r\\nthis deal is t...   \n",
              "\n",
              "   label_num  \n",
              "0          0  \n",
              "1          0  \n",
              "2          0  \n",
              "3          1  \n",
              "4          0  "
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-9b14425f-6058-4610-8515-e6b5aa36ad70\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>label</th>\n",
              "      <th>text</th>\n",
              "      <th>label_num</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>605</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: enron methanol ; meter # : 988291\\r\\n...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2349</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: hpl nom for january 9 , 2001\\r\\n( see...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3624</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: neon retreat\\r\\nho ho ho , we ' re ar...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4685</td>\n",
              "      <td>spam</td>\n",
              "      <td>Subject: photoshop , windows , office . cheap ...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2030</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: re : indian springs\\r\\nthis deal is t...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9b14425f-6058-4610-8515-e6b5aa36ad70')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-663cd85c-1de8-4cc0-91cc-eb29e69e5477\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-663cd85c-1de8-4cc0-91cc-eb29e69e5477')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-663cd85c-1de8-4cc0-91cc-eb29e69e5477 button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9b14425f-6058-4610-8515-e6b5aa36ad70 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9b14425f-6058-4610-8515-e6b5aa36ad70');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.tail()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "b_pmCdYHTxze",
        "outputId": "02118969-53f1-4a49-900b-cc4cab6635d0"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      Unnamed: 0 label                                               text  \\\n",
              "5166        1518   ham  Subject: put the 10 on the ft\\r\\nthe transport...   \n",
              "5167         404   ham  Subject: 3 / 4 / 2000 and following noms\\r\\nhp...   \n",
              "5168        2933   ham  Subject: calpine daily gas nomination\\r\\n>\\r\\n...   \n",
              "5169        1409   ham  Subject: industrial worksheets for august 2000...   \n",
              "5170        4807  spam  Subject: important online banking alert\\r\\ndea...   \n",
              "\n",
              "      label_num  \n",
              "5166          0  \n",
              "5167          0  \n",
              "5168          0  \n",
              "5169          0  \n",
              "5170          1  "
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-d181b570-5136-47be-a9ec-fe72529558e9\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>label</th>\n",
              "      <th>text</th>\n",
              "      <th>label_num</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>5166</th>\n",
              "      <td>1518</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: put the 10 on the ft\\r\\nthe transport...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5167</th>\n",
              "      <td>404</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: 3 / 4 / 2000 and following noms\\r\\nhp...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5168</th>\n",
              "      <td>2933</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: calpine daily gas nomination\\r\\n&gt;\\r\\n...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5169</th>\n",
              "      <td>1409</td>\n",
              "      <td>ham</td>\n",
              "      <td>Subject: industrial worksheets for august 2000...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5170</th>\n",
              "      <td>4807</td>\n",
              "      <td>spam</td>\n",
              "      <td>Subject: important online banking alert\\r\\ndea...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d181b570-5136-47be-a9ec-fe72529558e9')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-29542aa6-f98e-4ea9-9225-8555873e3dcd\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-29542aa6-f98e-4ea9-9225-8555873e3dcd')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-29542aa6-f98e-4ea9-9225-8555873e3dcd button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d181b570-5136-47be-a9ec-fe72529558e9 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d181b570-5136-47be-a9ec-fe72529558e9');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 5: Data Preprocessing\n",
        "# Split the dataset into features (text) and target (label)\n",
        "X = data['text']\n",
        "y = data['label']"
      ],
      "metadata": {
        "id": "uSUiOCYkUB1u"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6: Feature Engineering\n",
        "# Convert text data to numerical representations using CountVectorizer\n",
        "vectorizer = CountVectorizer()\n",
        "X = vectorizer.fit_transform(X)"
      ],
      "metadata": {
        "id": "OiOf2AtxUdrU"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6 (Continued): Splitting the dataset\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "WWL6PSd2Udxm"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6 (Continued): Model Creation and Training\n",
        "# Create a Naive Bayes classifier and train it\n",
        "model = MultinomialNB()\n",
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "dYkN9K2aUd09",
        "outputId": "f9b2d990-ee4c-46af-e60b-ffe7b02ff79b"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MultinomialNB()"
            ],
            "text/html": [
              "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6 (Continued): Making Predictions\n",
        "# Make predictions on the test set\n",
        "y_pred = model.predict(X_test)"
      ],
      "metadata": {
        "id": "TSkzI0DdUd3-"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 6 (Continued): Evaluation\n",
        "# Calculate the accuracy of the model\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f\"Accuracy: {accuracy:.2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vcc-bZomUpgl",
        "outputId": "c858f81d-6d31-4d6d-8e89-bf3d6b120167"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.98\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n25TwHnAUsOG",
        "outputId": "1d40d80a-a238-419d-8dd3-7c1e27daeed8"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 5171 entries, 0 to 5170\n",
            "Data columns (total 4 columns):\n",
            " #   Column      Non-Null Count  Dtype \n",
            "---  ------      --------------  ----- \n",
            " 0   Unnamed: 0  5171 non-null   int64 \n",
            " 1   label       5171 non-null   object\n",
            " 2   text        5171 non-null   object\n",
            " 3   label_num   5171 non-null   int64 \n",
            "dtypes: int64(2), object(2)\n",
            "memory usage: 161.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "E6UKP53FWGM2"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.pie(y.value_counts(), labels=['ham', 'spam'], autopct='%0.2f%%', explode=[0.1, 0])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 406
        },
        "id": "EIxdpUfvWIX9",
        "outputId": "3a1a7d70-1c46-43f9-e1cb-9e9d88676700"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "value_counts = y.value_counts()\n",
        "\n",
        "# Get the categories ('ham' and 'spam') and their corresponding counts\n",
        "categories = value_counts.index\n",
        "counts = value_counts.values\n",
        "\n",
        "# Create the bar graph\n",
        "plt.bar(categories, counts)\n",
        "\n",
        "# Set labels and title\n",
        "plt.xlabel('Categories')\n",
        "plt.ylabel('Count')\n",
        "plt.title('Category Counts')\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "wv76qqL-XFIt",
        "outputId": "d04a8fd5-8809-4bb3-cec9-cff9302eebcd"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHHCAYAAABeLEexAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA9cklEQVR4nO3deVhV5f7//xcgIKAbRGUwEQdyQEXLysgsS5QULcs62eBQampgJyk1zsfj1EBHc8ocjvVJ7BxttklyhMhUUsMPOaSmHs1OMjjBdgSF+/dHP/a3nUNi4EbX83Fd67pc9/1ea9335iJerWm7GWOMAAAALMzd1QMAAABwNQIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRYGF79uzRkCFD1LhxY1WvXl02m00dOnTQjBkzdOrUqXLvb/bs2UpJSan4gVZhFf0ZVgYr/lyA8nLju8wAa0pNTdVDDz0kb29v9evXT61atVJxcbHWrFmjjz/+WAMGDNC8efPKtc9WrVqpTp06ysjIqJxBVzGV8RlWBqv9XIDLUc3VAwBw5e3du1d9+vRReHi40tPTFRoa6uiLj4/X7t27lZqa6sIRVq4TJ07Iz8/vT+3D6p8hcM0xACxn6NChRpJZu3btJdW//fbb5q677jJ169Y1Xl5epkWLFmb27NlONeHh4UaS03LnnXc6+o8ePWr++te/mvr16xsvLy/TpEkT8+qrr5qSkhKn/Rw6dMg8/vjjpmbNmsbf39/069fPZGdnG0lm/vz5TrVpaWnm9ttvN76+vsbf39/ce++95ocffnCqGTdunJFktm3bZh555BETEBBg2rZta95++20jyWzatOmc+b788svG3d3d/Pe//62wz/DMmTNm4sSJpnHjxsbLy8uEh4ebpKQkc/r0aac6SWbcuHHnbB8eHm769+/vWJ8/f76RZNasWWNGjBhh6tSpY3x9fU2vXr1Mfn6+03YX+rkUFxeb8ePHm4iICOPt7W0CAwNNhw4dzIoVKy5pTsC1hDNEgAV98cUXaty4sW677bZLqp8zZ45atmype++9V9WqVdMXX3yhp59+WqWlpYqPj5ckTZ8+XcOHD1eNGjX0P//zP5Kk4OBgSdLJkyd155136pdfftGQIUPUoEEDrVu3TklJScrJydH06dMlSaWlperZs6c2bNigYcOGqXnz5vrss8/Uv3//c8a0atUqdevWTY0bN9b48eN16tQpzZw5Ux06dNCmTZvUsGFDp/qHHnpI119/vV555RUZY/Tggw8qPj5eCxcu1A033OBUu3DhQnXq1EnXXXddhX2GgwYN0oIFC/Tggw/queee0/r165WcnKzt27frk08+uaR9nM/w4cNVq1YtjRs3Tvv27dP06dOVkJCg999/X9LFfy7jx49XcnKyBg0apFtuuUV2u13fffedNm3apC5dulz2mICrkqsTGYArq7Cw0Egy99133yVvc/LkyXPaYmNjTePGjZ3aWrZs6XRWqMyLL75o/Pz8zI8//ujU/sILLxgPDw+zf/9+Y4wxH3/8sZFkpk+f7qgpKSkxd9999zlniNq2bWuCgoLM4cOHHW3ff/+9cXd3N/369XO0lZ0heuSRR84Z1yOPPGLq1avndJZq06ZN5z0b9Vvl/QzLznANGjTIqf355583kkx6erqjTeU8QxQTE2NKS0sd7SNGjDAeHh6moKDA0Xahn0ubNm1MXFzcJc0BuNbxlBlgMXa7XZJUs2bNS97Gx8fH8e/CwkIdOnRId955p/7zn/+osLDwD7f/8MMP1bFjR9WqVUuHDh1yLDExMSopKdHq1aslScuWLZOnp6cGDx7s2Nbd3d1xFqpMTk6OsrOzNWDAAAUGBjrao6Ki1KVLF3355ZfnjGHo0KHntPXr108HDhzQV1995WhbuHChfHx81Lt37wvOp7yfYdl4EhMTndqfe+45SfpT9xo99dRTcnNzc6x37NhRJSUl+umnn/5w24CAAG3btk27du267OMD1woCEWAxNptNknTs2LFL3mbt2rWKiYmRn5+fAgICVLduXf3tb3+TpEsKRLt27dKyZctUt25dpyUmJkaSlJ+fL0n66aefFBoaKl9fX6ftIyIinNbL/tg3a9bsnGO1aNFChw4d0okTJ5zaGzVqdE5tly5dFBoaqoULF0r69ZLdu+++q/vuu++iYae8n+FPP/0kd3f3c+YREhKigICASwovF9KgQQOn9Vq1akmSjh49+ofbTpw4UQUFBWratKlat26tkSNHavPmzZc9FuBqxj1EgMXYbDbVq1dPW7duvaT6PXv2qHPnzmrevLmmTp2qsLAweXl56csvv9S0adNUWlr6h/soLS1Vly5dNGrUqPP2N23atFxzuBy/PctVxsPDQ48++qjefPNNzZ49W2vXrtWBAwf0+OOPX3Rf5f0My/z2TE55lZSUnLfdw8PjvO3mEt6ocscdd2jPnj367LPPtGLFCr311luaNm2a5s6dq0GDBl32WIGrEWeIAAvq0aOH9uzZo8zMzD+s/eKLL1RUVKTPP/9cQ4YMUffu3RUTE3PegHGhP/hNmjTR8ePHFRMTc96l7CxHeHi4cnJydPLkSaftd+/e7bQeHh4uSdq5c+c5x9qxY4fq1KlzyY/V9+vXT3a7XV988YUWLlyounXrKjY29g+3K89nGB4ertLS0nMuTeXl5amgoMAxH+nXMzwFBQVOdcXFxcrJybmk+ZzPxYJYYGCgnnjiCb377rv6+eefFRUVpfHjx1/2sYCrFYEIsKBRo0bJz89PgwYNUl5e3jn9e/bs0YwZMyT9vzMQvz3jUFhYqPnz55+znZ+f3zl/zCXpL3/5izIzM7V8+fJz+goKCnT27FlJUmxsrM6cOaM333zT0V9aWqpZs2Y5bRMaGqq2bdtqwYIFTsfbunWrVqxYoe7du19k9s6ioqIUFRWlt956Sx9//LH69OmjatX++OR5eT7DsvGUPU1XZurUqZKkuLg4R1uTJk0c91SVmTdv3gXPEF2KC/1cDh8+7LReo0YNRUREqKio6LKPBVytuGQGWFCTJk20aNEiPfzww2rRooXTW5bXrVunDz/8UAMGDJAkde3aVV5eXurZs6eGDBmi48eP680331RQUNA5Zy3atWunOXPm6KWXXlJERISCgoJ09913a+TIkfr888/Vo0cPDRgwQO3atdOJEye0ZcsWffTRR9q3b5/q1KmjXr166ZZbbtFzzz2n3bt3q3nz5vr888915MgRSc5nOiZPnqxu3bopOjpaAwcOdDx27+/vX+4zHP369dPzzz8vSX94uexyPsM2bdqof//+mjdvngoKCnTnnXdqw4YNWrBggXr16qW77rrLsd9BgwZp6NCh6t27t7p06aLvv/9ey5cvV506dco1p9+60M8lMjJSnTp1Urt27RQYGKjvvvtOH330kRISEi77WMBVy9WPuQFwnR9//NEMHjzYNGzY0Hh5eZmaNWuaDh06mJkzZzq9MPDzzz83UVFRpnr16qZhw4bmH//4h+PFhnv37nXU5ebmmri4OFOzZs1zXsx47Ngxk5SUZCIiIoyXl5epU6eOue2228xrr71miouLHXUHDx40jz76qOPFjAMGDDBr1641ksx7773nNP5Vq1aZDh06GB8fH2Oz2UzPnj0v+GLGgwcPXvBzyMnJMR4eHqZp06aV9hmeOXPGTJgwwTRq1Mh4enqasLCw876YsaSkxIwePdrxosXY2Fize/fuCz52v3HjRqftv/rqKyPJfPXVV462C/1cXnrpJXPLLbeYgIAA4+PjY5o3b25efvllp58HYBV8lxmAKu/TTz/V/fffrzVr1qhDhw4Vvv9Dhw4pNDRUY8eO1d///vcK3z+Aqo97iABUKb//hviSkhLNnDlTNptNN954Y6UcMyUlRSUlJerbt2+l7B9A1cc9RACqlOHDh+vUqVOKjo5WUVGRFi9erHXr1umVV14575Ntf0Z6erp++OEHvfzyy+rVq9c5X/cBwDq4ZAagSlm0aJGmTJmi3bt36/Tp04qIiNCwYcMq5UbfTp06ad26derQoYP+/e9/X/S7ywBc2whEAADA8riHCAAAWB6BCAAAWB43VV+C0tJSHThwQDVr1vxT30UEAACuHGOMjh07pnr16snd/eLngAhEl+DAgQMKCwtz9TAAAMBl+Pnnn1W/fv2L1hCILkHNmjUl/fqB2mw2F48GAABcCrvdrrCwMMff8YshEF2CsstkNpuNQAQAwFXmUm534aZqAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgedVcPQBIDV9IdfUQgCpr36txrh4CAAvgDBEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8AhEAALA8lwaiOXPmKCoqSjabTTabTdHR0Vq6dKmjv1OnTnJzc3Nahg4d6rSP/fv3Ky4uTr6+vgoKCtLIkSN19uxZp5qMjAzdeOON8vb2VkREhFJSUq7E9AAAwFXCpV/uWr9+fb366qu6/vrrZYzRggULdN999+n//u//1LJlS0nS4MGDNXHiRMc2vr6+jn+XlJQoLi5OISEhWrdunXJyctSvXz95enrqlVdekSTt3btXcXFxGjp0qBYuXKi0tDQNGjRIoaGhio2NvbITBgAAVZKbMca4ehC/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Ex3rdrtdYWFhlXIsAADgelXikllCQoKWLFmir776SvXr179obfv27SVJu3fvliSFhIQoLy/PqaZsvey+owvV2Gy2c84OSZK3t7dsNpvTAgAArl0uDUTGGCUkJOiTTz5Renq6GjVq9IfbZGdnS5JCQ0MlSdHR0dqyZYvy8/MdNStXrpTNZlNkZKSjJi0tzWk/K1euVHR0dAXNBAAAXM1cGoji4+P173//W4sWLVLNmjWVm5ur3NxcnTp1SpK0Z88evfjii8rKytK+ffv0+eefq1+/frrjjjsUFRUlSeratasiIyPVt29fff/991q+fLnGjBmj+Ph4eXt7S5KGDh2q//znPxo1apR27Nih2bNn64MPPtCIESNcNncAAFB1uDQQzZkzR4WFherUqZNCQ0Mdy/vvvy9J8vLy0qpVq9S1a1c1b95czz33nHr37q0vvvjCsQ8PDw8tWbJEHh4eio6O1uOPP65+/fpp4sSJjppGjRopNTVVK1euVJs2bTRlyhS99dZbPHIPAAAkSW7GGOPqQVR1drtd/v7+KiwsrJT7iRq+kFrh+wSuFftejXP1EABcpcrz97tK3FQNAADgSgQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeQQiAABgeZcViBo3bqzDhw+f015QUKDGjRv/6UEBAABcSZcViPbt26eSkpJz2ouKivTLL7/86UEBAABcSdXKU/z55587/r18+XL5+/s71ktKSpSWlqaGDRtW2OAAAACuhHIFol69ekmS3Nzc1L9/f6c+T09PNWzYUFOmTKmwwQEAAFwJ5QpEpaWlkqRGjRpp48aNqlOnTqUMCgAA4EoqVyAqs3fv3ooeBwAAgMtcViCSpLS0NKWlpSk/P99x5qjM22+//acHBgAAcKVc1lNmEyZMUNeuXZWWlqZDhw7p6NGjTsulSk5O1s0336yaNWsqKChIvXr10s6dO51qTp8+rfj4eNWuXVs1atRQ7969lZeX51Szf/9+xcXFydfXV0FB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SUpMTHSs2+12QhEAANewKn2G6PcCAgLUtGlT7d69WyEhISouLlZBQYFTTV5enuOeo5CQkHOeOitbP999SWW8vb1ls9mcFgAAcO26qgLR8ePHtWfPHoWGhqpdu3by9PRUWlqao3/nzp3av3+/oqOjJUnR0dHasmWL8vPzHTUrV66UzWZTZGTkFR8/AAComqr0JbPnn39ePXv2VHh4uA4cOKBx48bJw8NDjzzyiPz9/TVw4EAlJiYqMDBQNptNw4cPV3R0tG699VZJUteuXRUZGam+fftq0qRJys3N1ZgxYxQfHy9vb28Xzw4AAFQVVToQ/fe//9Ujjzyiw4cPq27durr99tv17bffqm7dupKkadOmyd3dXb1791ZRUZFiY2M1e/Zsx/YeHh5asmSJhg0bpujoaPn5+al///6aOHGiq6YEAACqoCodiN57772L9levXl2zZs3SrFmzLlgTHh6uL7/8sqKHBgAAriFX1T1EAAAAlYFABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALM9SgWjWrFlq2LChqlevrvbt22vDhg2uHhIAAKgCLBOI3n//fSUmJmrcuHHatGmT2rRpo9jYWOXn57t6aAAAwMUsE4imTp2qwYMH64knnlBkZKTmzp0rX19fvf32264eGgAAcDFLBKLi4mJlZWUpJibG0ebu7q6YmBhlZma6cGQAAKAqqObqAVwJhw4dUklJiYKDg53ag4ODtWPHjnPqi4qKVFRU5FgvLCyUJNnt9koZX2nRyUrZL3AtqKzfuyuN33Pg4irjd71sn8aYP6y1RCAqr+TkZE2YMOGc9rCwMBeMBrA2/+muHgGAK6Eyf9ePHTsmf3//i9ZYIhDVqVNHHh4eysvLc2rPy8tTSEjIOfVJSUlKTEx0rJeWlurIkSOqXbu23NzcKn28cB273a6wsDD9/PPPstlsrh4OgErC77o1GGN07Ngx1atX7w9rLRGIvLy81K5dO6WlpalXr16Sfg05aWlpSkhIOKfe29tb3t7eTm0BAQFXYKSoKmw2G/+RBCyA3/Vr3x+dGSpjiUAkSYmJierfv79uuukm3XLLLZo+fbpOnDihJ554wtVDAwAALmaZQPTwww/r4MGDGjt2rHJzc9W2bVstW7bsnButAQCA9VgmEElSQkLCeS+RAWW8vb01bty4cy6ZAri28LuO33Mzl/IsGgAAwDXMEi9mBAAAuBgCEQAAsDwCEQAAsDwCEa5ZnTp10rPPPuvqYQAArgIEIgAAYHkEIgAAYHkEIlzTSktLNWrUKAUGBiokJETjx4939E2dOlWtW7eWn5+fwsLC9PTTT+v48eOO/pSUFAUEBGjJkiVq1qyZfH199eCDD+rkyZNasGCBGjZsqFq1aumZZ55RSUmJC2YHWNNHH32k1q1by8fHR7Vr11ZMTIxOnDihAQMGqFevXpowYYLq1q0rm82moUOHqri42LHtsmXLdPvttysgIEC1a9dWjx49tGfPHkf/vn375Obmpg8++EAdO3aUj4+Pbr75Zv3444/auHGjbrrpJtWoUUPdunXTwYMHXTF9VBICEa5pCxYskJ+fn9avX69JkyZp4sSJWrlypSTJ3d1dr7/+urZt26YFCxYoPT1do0aNctr+5MmTev311/Xee+9p2bJlysjI0P33368vv/xSX375pf71r3/pn//8pz766CNXTA+wnJycHD3yyCN68skntX37dmVkZOiBBx5Q2Sv10tLSHO3vvvuuFi9erAkTJji2P3HihBITE/Xdd98pLS1N7u7uuv/++1VaWup0nHHjxmnMmDHatGmTqlWrpkcffVSjRo3SjBkz9M0332j37t0aO3bsFZ07KpkBrlF33nmnuf32253abr75ZjN69Ojz1n/44Yemdu3ajvX58+cbSWb37t2OtiFDhhhfX19z7NgxR1tsbKwZMmRIBY8ewPlkZWUZSWbfvn3n9PXv398EBgaaEydOONrmzJljatSoYUpKSs67v4MHDxpJZsuWLcYYY/bu3WskmbfeestR8+677xpJJi0tzdGWnJxsmjVrVlHTQhXAGSJc06KiopzWQ0NDlZ+fL0latWqVOnfurOuuu041a9ZU3759dfjwYZ08edJR7+vrqyZNmjjWg4OD1bBhQ9WoUcOprWyfACpXmzZt1LlzZ7Vu3VoPPfSQ3nzzTR09etSp39fX17EeHR2t48eP6+eff5Yk7dq1S4888ogaN24sm82mhg0bSpL279/vdJzf/rej7DsvW7du7dTG7/21hUCEa5qnp6fTupubm0pLS7Vv3z716NFDUVFR+vjjj5WVlaVZs2ZJktP9Bufb/kL7BFD5PDw8tHLlSi1dulSRkZGaOXOmmjVrpr17917S9j179tSRI0f05ptvav369Vq/fr0k5997yfl3383N7bxt/N5fWyz15a5AmaysLJWWlmrKlClyd//1/ws++OADF48KwKVwc3NThw4d1KFDB40dO1bh4eH65JNPJEnff/+9Tp06JR8fH0nSt99+qxo1aigsLEyHDx/Wzp079eabb6pjx46SpDVr1rhsHqhaCESwpIiICJ05c0YzZ85Uz549tXbtWs2dO9fVwwLwB9avX6+0tDR17dpVQUFBWr9+vQ4ePKgWLVpo8+bNKi4u1sCBAzVmzBjt27dP48aNU0JCgtzd3VWrVi3Vrl1b8+bNU2hoqPbv368XXnjB1VNCFcElM1hSmzZtNHXqVP3jH/9Qq1attHDhQiUnJ7t6WAD+gM1m0+rVq9W9e3c1bdpUY8aM0ZQpU9StWzdJUufOnXX99dfrjjvu0MMPP6x7773X8boNd3d3vffee8rKylKrVq00YsQITZ482YWzQVXiZsz//6wiAABXsQEDBqigoECffvqpq4eCqxBniAAAgOURiAAAgOVxyQwAAFgeZ4gAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAoAJlZGTIzc1NBQUFrh4KgHIgEAFwmdzcXA0fPlyNGzeWt7e3wsLC1LNnT6WlpV3S9ikpKQoICKjcQZbTbbfdppycHPn7+7t6KADKge8yA+AS+/btU4cOHRQQEKDJkyerdevWOnPmjJYvX674+Hjt2LHD1UMstzNnzsjLy0shISGuHgqAcuIMEQCXePrpp+Xm5qYNGzaod+/eatq0qVq2bKnExER9++23kqSpU6eqdevW8vPzU1hYmJ5++mkdP35c0q+Xpp544gkVFhbKzc1Nbm5uju+sKioq0vPPP6/rrrtOfn5+at++vTIyMpyO/+abbyosLEy+vr66//77NXXq1HPONs2ZM0dNmjSRl5eXmjVrpn/9619O/W5ubpozZ47uvfde+fn56eWXXz7vJbM1a9aoY8eO8vHxUVhYmJ555hmdOHHC0T979mxdf/31ql69uoKDg/Xggw9WzIcM4NIZALjCDh8+bNzc3Mwrr7xy0bpp06aZ9PR0s3fvXpOWlmaaNWtmhg0bZowxpqioyEyfPt3YbDaTk5NjcnJyzLFjx4wxxgwaNMjcdtttZvXq1Wb37t1m8uTJxtvb2/z444/GGGPWrFlj3N3dzeTJk83OnTvNrFmzTGBgoPH393cce/HixcbT09PMmjXL7Ny500yZMsV4eHiY9PR0R40kExQUZN5++22zZ88e89NPP5mvvvrKSDJHjx41xhize/du4+fnZ6ZNm2Z+/PFHs3btWnPDDTeYAQMGGGOM2bhxo/Hw8DCLFi0y+/btM5s2bTIzZsyoqI8awCUiEAG44tavX28kmcWLF5druw8//NDUrl3bsT5//nynEGOMMT/99JPx8PAwv/zyi1N7586dTVJSkjHGmIcfftjExcU59T/22GNO+7rtttvM4MGDnWoeeugh0717d8e6JPPss8861fw+EA0cONA89dRTTjXffPONcXd3N6dOnTIff/yxsdlsxm63//EHAKDScMkMwBVnLvEbg1atWqXOnTvruuuuU82aNdW3b18dPnxYJ0+evOA2W7ZsUUlJiZo2baoaNWo4lq+//lp79uyRJO3cuVO33HKL03a/X9++fbs6dOjg1NahQwdt377dqe2mm2666By+//57paSkOI0lNjZWpaWl2rt3r7p06aLw8HA1btxYffv21cKFCy86PwCVg5uqAVxx119/vdzc3C564/S+ffvUo0cPDRs2TC+//LICAwO1Zs0aDRw4UMXFxfL19T3vdsePH5eHh4eysrLk4eHh1FejRo0KnYck+fn5XbT/+PHjGjJkiJ555plz+ho0aCAvLy9t2rRJGRkZWrFihcaOHavx48dr48aNVe4JOuBaxhkiAFdcYGCgYmNjNWvWLKebi8sUFBQoKytLpaWlmjJlim699VY1bdpUBw4ccKrz8vJSSUmJU9sNN9ygkpIS5efnKyIiwmkpe/qrWbNm2rhxo9N2v19v0aKF1q5d69S2du1aRUZGlmuuN954o3744YdzxhIRESEvLy9JUrVq1RQTE6NJkyZp8+bN2rdvn9LT08t1HAB/DoEIgEvMmjVLJSUluuWWW/Txxx9r165d2r59u15//XVFR0crIiJCZ86c0cyZM/Wf//xH//rXvzR37lynfTRs2FDHjx9XWlqaDh06pJMnT6pp06Z67LHH1K9fPy1evFh79+7Vhg0blJycrNTUVEnS8OHD9eWXX2rq1KnatWuX/vnPf2rp0qVyc3Nz7HvkyJFKSUnRnDlztGvXLk2dOlWLFy/W888/X655jh49WuvWrVNCQoKys7O1a9cuffbZZ0pISJAkLVmyRK+//rqys7P1008/6Z133lFpaamaNWv2Jz9hAOXi6puYAFjXgQMHTHx8vAkPDzdeXl7muuuuM/fee6/56quvjDHGTJ061YSGhhofHx8TGxtr3nnnHacblo0xZujQoaZ27dpGkhk3bpwxxpji4mIzduxY07BhQ+Pp6WlCQ0PN/fffbzZv3uzYbt68eea6664zPj4+plevXuall14yISEhTuObPXu2ady4sfH09DRNmzY177zzjlO/JPPJJ584tf3+pmpjjNmwYYPp0qWLqVGjhvHz8zNRUVHm5ZdfNsb8eoP1nXfeaWrVqmV8fHxMVFSUef/99//cBwug3NyMucS7GwHgGjZ48GDt2LFD33zzjauHAsAFuKkagCW99tpr6tKli/z8/LR06VItWLBAs2fPdvWwALgIZ4gAWNJf/vIXZWRk6NixY2rcuLGGDx+uoUOHunpYAFyEQAQAACyPp8wAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAAIDl/X+6Ak7end4LdwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate x-values from 0 to 2*pi (full circle) with small increments\n",
        "x = np.linspace(0, 2 * np.pi, 100)\n",
        "\n",
        "# Calculate y-values for sine function\n",
        "y = np.sin(x)\n",
        "\n",
        "# Create the trigonometric graph\n",
        "plt.plot(x, y)\n",
        "\n",
        "# Set labels and title\n",
        "plt.xlabel('x')\n",
        "plt.ylabel('sin(x)')\n",
        "plt.title('Sine Graph')\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "jv7nzyKdXNK9",
        "outputId": "42436c46-b074-45a2-819f-3593ccfcda6b"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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EREREWnBYomIiIhIC71Z7kSX3WuCLpfLBU5CRERE9XXve/thi5mwWGoCRUVFAAB3d3eBkxAREdGjKioqgkwme+DrXBuuCSiVSmRkZMDKygoikajJriuXy+Hu7o60tDSuOVcHfj7a8fN5MH422vHz0Y6fj3b69PmoVCoUFRXB1dUVYvGDZyZxZKkJiMVitG3bttmub21trfP/wgmJn492/HwejJ+Ndvx8tOPno52+fD7aRpTu4QRvIiIiIi1YLBERERFpwWJJh0mlUsyfPx9SqVToKDqJn492/HwejJ+Ndvx8tOPno50hfj6c4E1ERESkBUeWiIiIiLRgsURERESkBYslIiIiIi1YLBERERFpwWJJhy1fvhyenp4wNTVFQEAAoqKihI6kE06ePImnn34arq6uEIlE2LNnj9CRdMaiRYvQt29fWFlZwdHREePGjcP169eFjqUzVqxYgR49emia5QUGBuLgwYNCx9JZX331FUQiEebMmSN0FJ3w6aefQiQS1dp8fHyEjqUz0tPT8fLLL6NNmzYwMzND9+7dcfHiRaFjNQkWSzpqx44dmDt3LubPn4+YmBj4+fkhODgYOTk5QkcTXElJCfz8/LB8+XKho+icEydOYObMmTh37hyOHDmCqqoqDB8+HCUlJUJH0wlt27bFV199hejoaFy8eBFPPPEExo4di6tXrwodTedcuHABq1atQo8ePYSOolO6du2KzMxMzXb69GmhI+mEu3fvYuDAgTA2NsbBgwdx7do1fPfdd7C1tRU6WpNg6wAdFRAQgL59++KHH34AoF5/zt3dHbNnz8aHH34ocDrdIRKJsHv3bowbN07oKDopNzcXjo6OOHHiBIYMGSJ0HJ1kZ2eHb775BlOnThU6is4oLi5Gr1698N///hdffPEF/P39sXTpUqFjCe7TTz/Fnj17EBsbK3QUnfPhhx/izJkzOHXqlNBRmgVHlnRQZWUloqOjERQUpNknFosRFBSEyMhIAZORviksLASgLgioNoVCge3bt6OkpASBgYFCx9EpM2fOxOjRo2v9GURqN27cgKurK7y9vTFx4kSkpqYKHUkn7N27F3369MHzzz8PR0dH9OzZE2vWrBE6VpNhsaSD8vLyoFAo4OTkVGu/k5MTsrKyBEpF+kapVGLOnDkYOHAgunXrJnQcnXHlyhVYWlpCKpXijTfewO7du+Hr6yt0LJ2xfft2xMTEYNGiRUJH0TkBAQHYuHEjwsPDsWLFCiQnJ2Pw4MEoKioSOprgbt68iRUrVqBjx444dOgQZsyYgbfeegubNm0SOlqTMBI6ABE1j5kzZyIuLo5zKv6hc+fOiI2NRWFhIX755RdMnjwZJ06cYMEEIC0tDW+//TaOHDkCU1NToePonJEjR2r+f48ePRAQEAAPDw/8/PPPrf42rlKpRJ8+fbBw4UIAQM+ePREXF4eVK1di8uTJAqdrPI4s6SB7e3tIJBJkZ2fX2p+dnQ1nZ2eBUpE+mTVrFvbv349jx46hbdu2QsfRKSYmJujQoQN69+6NRYsWwc/PD//5z3+EjqUToqOjkZOTg169esHIyAhGRkY4ceIEvv/+exgZGUGhUAgdUafY2NigU6dOSExMFDqK4FxcXO77C0eXLl0M5jYliyUdZGJigt69eyMiIkKzT6lUIiIignMrSCuVSoVZs2Zh9+7dOHr0KLy8vISOpPOUSiUqKiqEjqEThg0bhitXriA2Nlaz9enTBxMnTkRsbCwkEonQEXVKcXExkpKS4OLiInQUwQ0cOPC+NiV//vknPDw8BErUtHgbTkfNnTsXkydPRp8+fdCvXz8sXboUJSUlCA0NFTqa4IqLi2v9TS45ORmxsbGws7NDu3btBEwmvJkzZ2Lbtm343//+BysrK80cN5lMBjMzM4HTCS8sLAwjR45Eu3btUFRUhG3btuH48eM4dOiQ0NF0gpWV1X3z2ywsLNCmTRvOewPwr3/9C08//TQ8PDyQkZGB+fPnQyKRYMKECUJHE9w777yDAQMGYOHChXjhhRcQFRWF1atXY/Xq1UJHaxoq0lnLli1TtWvXTmViYqLq16+f6ty5c0JH0gnHjh1TAbhvmzx5stDRBFfX5wJAtWHDBqGj6YRXX31V5eHhoTIxMVE5ODiohg0bpjp8+LDQsXTaY489pnr77beFjqETXnzxRZWLi4vKxMRE5ebmpnrxxRdViYmJQsfSGfv27VN169ZNJZVKVT4+PqrVq1cLHanJsM8SERERkRacs0RERESkBYslIiIiIi1YLBERERFpwWKJiIiISAsWS0RERERasFgiIiIi0oLFEhEREZEWLJaIiIiItGCxRERERKQFiyUiIiIiLVgsERH9Q25uLpydnbFw4ULNvrNnz8LExAQRERECJiMiIXBtOCKiOhw4cADjxo3D2bNn0blzZ/j7+2Ps2LFYsmSJ0NGIqIWxWCIieoCZM2fi999/R58+fXDlyhVcuHABUqlU6FhE1MJYLBERPUBZWRm6deuGtLQ0REdHo3v37kJHIiIBcM4SEdEDJCUlISMjA0qlEikpKULHISKBcGSJiKgOlZWV6NevH/z9/dG5c2csXboUV65cgaOjo9DRiKiFsVgiIqrDe++9h19++QWXLl2CpaUlHnvsMchkMuzfv1/oaETUwngbjojoH44fP46lS5fixx9/hLW1NcRiMX788UecOnUKK1asEDoeEbUwjiwRERERacGRJSIiIiItWCwRERERacFiiYiIiEgLFktEREREWrBYIiIiItKCxRIRERGRFiyWiIiIiLRgsURERESkBYslIiIiIi1YLBERERFpwWKJiIiISAsWS0RERERa/D+eiV4AGuydEAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate x-values from 0 to 2*pi (full circle) with small increments\n",
        "x = np.linspace(0, 2 * np.pi, 100)\n",
        "x = np.linspace(0, 2 * np.pi, 100)\n",
        "\n",
        "# Calculate y-values for sine function\n",
        "y = np.sin(x)\n",
        "z = np.cos(x)\n",
        "# Create the trigonometric graph\n",
        "plt.plot(x, y)\n",
        "plt.plot(z)\n",
        "\n",
        "# Set labels and title\n",
        "plt.xlabel('y')\n",
        "plt.ylabel('sin(x) and Cos(x)')\n",
        "plt.title('The graph the trigonometrical Graphs')\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "ar4lDYgnY9oo",
        "outputId": "78b03e5c-9d40-43ea-af1f-8f4877725569"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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S5a0+FrXqVOH9957j3HjxjF8+HAeeughAFq1amWKIa/fz7ulp6fz6KOPsn79evr168fLL7/MrVu3iIyM5PDhw6af1eeff063bt0YOHAgKSkpLF26lD59+rBq1apcfwb5+e2334CcLcyWmDRpEp6enrz22mskJyfj6enJkSNHWLFiBX369KFq1arExMQwe/Zs2rZty5EjRwgODjY7x+TJk9HpdLz55ptcuXKFadOmER4ezv79+/H29jYdd/PmTTp16kTPnj154okn+Omnn3jzzTdp0KABnTt3BmDu3Lm89NJL9O7dm5dffpmkpCQOHjzIjh07GDBggNWfT9iZIoQDGzlypJLXr/HGjRsVQKlTp46SnJxs2v/5558rgHLo0CFFURQlIyNDqVmzphIREaFkZGSYjktMTFSqVq2qPPLII/nG8OmnnyqAsmLFCtO+O3fuKLVr11YAZePGjab9bdu2VQBl1qxZOc6TmJiYY99zzz2n+Pj4KElJSTnO8emnn5r2JScnK40bN1bKlSunpKSkWPX58/Lxxx8rgHL69OkcrwGKp6encvLkSdO+AwcOKIAyffp0076hQ4cqFSpUUK5du2b2/n79+in+/v65fubsSpQooQwePDjH/vHjxyuA0r9//zxfM9qzZ48CKKNHjzY77umnn1YAZfz48aZ9PXr0UDw9PZX//vvPtO/SpUtKyZIllTZt2pj2LViwQAGU8PBws9+ZV155RXFzc1NiY2MVRVGUK1euKJ6enkrHjh2V9PR003FffvmlAijz58837TP+XJcsWWLad+zYMQVQ9Hq9sn37dtP+NWvWKICyYMEC0z5Lr/WuXbtyvPfuGHL7/Wzbtq3Stm1b0/P58+crgPLZZ5/lOPbuf0fZpaSkKPXr11cefvhhs/2VK1fO9Wed3eOPP64AputrdOfOHeXq1aumx82bN02vGf8dVKtWLUcsSUlJZj8XRVGU06dPKwaDQXnvvfdynOO+++5T4uPjTft/+OEHBVA+//xz0z7jNVy0aJFpX3JyshIUFKT06tXLtK979+5KvXr18v28oviQbjjh9IYMGWI23sL41/SpU6cA2L9/PydOnGDAgAFcv36da9euce3aNRISEujQoQN//fUXGRkZeZ5/9erV3HfffXTr1s20z8vLi2HDhuV6vMFgYMiQITn2Z//L9NatW1y7do2HHnqIxMRE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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}