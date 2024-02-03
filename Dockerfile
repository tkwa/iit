FROM pytorch/pytorch as base

ENV DEBIAN_FRONTEND=noninteractive

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo

# Install some useful packages
RUN apt update && \
    apt install -y rsync git vim nano curl wget htop tmux zip unzip iputils-ping openssh-server strace nodejs python3 python3-pip ffmpeg zstd gcc \
    && python3 -m pip install --upgrade --no-cache-dir pip requests

FROM base as prod
ADD . .
RUN python3 -m pip install "."

FROM base as dev
WORKDIR /workspace
ARG GH_TOKEN=$GH_TOKEN
RUN curl -fsSL https://code-server.dev/install.sh | sh
ENV PATH="/home/dev/.local/bin:${PATH}"
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension github.copilot
RUN code-server --install-extension github.copilot-chat
RUN code-server --install-extension ms-python.isort
RUN code-server --install-extension ms-toolsai.jupyter
RUN code-server --install-extension ms-python.black-formatter

# vscode tunnel
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz
# ./code tunnel --disable-telemetry --random-name --accept-server-license-terms

RUN git config --global credential.helper store
RUN type -p curl >/dev/null || (apt update && apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& apt update \
&& apt install gh -y
# set to PAT
RUN gh config set git_protocol https
RUN git clone https://${GH_TOKEN}@github.com/cybershiptrooper/iit

RUN pip install sparse_autoencoder==1.8 transformer_lens pytest
RUN pip install poetry