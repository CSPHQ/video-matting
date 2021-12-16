FROM python:3.8
RUN useradd -ms /bin/bash qhduan
USER qhduan
WORKDIR /home/qhduan
ENV PATH="/home/qhduan/.local/bin:${PATH}"
COPY --chown=qhduan:qhduan ./requirements.txt .
RUN pip install --upgrade --user pip -i https://mirrors.aliyun.com/pypi/simple
RUN pip install --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
COPY --chown=qhduan:qhduan ./rvm.py .
RUN python rvm.py
