# front/Dockerfile
FROM python:3.11

# define a variável de ambiente GRADIO_SERVER_PORT com o valor padrão 7860
ARG GRADIO_SERVER_PORT=7860

# torna a variável de ambiente GRADIO_SERVER_PORT acessível para o contêiner
# isso permite que o código dentro do contêiner acesse essa variável
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# define o diretório de trabalho dentro do contêiner
# todos os comandos subsequentes serão executados nesse diretório
WORKDIR /work 

# copia o arquivo requirements.txt e o app-gradio.py para o diretório de trabalho no contêiner
# isso garante que esses arquivos estejam disponíveis para instalação de dependências e execução do aplicativo
ADD requirements.txt app_gradio.py /work/

# instala as dependências do Python listadas no arquivo requirements.txt
RUN pip install -r /work/requirements.txt

# documenta o que o contêiner faz
# isso é útil para quem lê o Dockerfile e para ferramentas de documentação
EXPOSE 7860

# define o comando padrão a ser executado quando o contêiner é iniciado
# nesse caso, o comando executa o aplicativo Gradio usando Python
CMD ["python","/work/app_gradio.py"]
