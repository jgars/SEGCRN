# [Choice] PyTorch version
ARG VARIANT="2.1.0-cuda11.8-cudnn8-runtime"
FROM pytorch/pytorch:${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# [Optional] Install Git
RUN apt-get update && apt-get install -y git

# Enable safe.dictory in git
RUN git config --global --add safe.directory /workspace

# Enable bash autocompletion for root
RUN apt-get install -y bash-completion
RUN echo 'if [ -f /etc/bash_completion ]; then' >> /root/.bashrc
RUN echo '  . /etc/bash_completion' >> /root/.bashrc
RUN echo 'fi' >> /root/.bashrc

# Create a user to avoid saving all the files in the container as root
ARG USERNAME=ai
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash -d /home/$USERNAME/ --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support 
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Add project to Python path (non-root)
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/"' >> ~/.bashrc
