import os
import shutil


def re_arrange_files(source='./', destination='./audio_files/'):
    try:
        os.mkdir(destination)
    except FileExistsError:
        pass

    try:
        shutil.rmtree('./sample_data/')
    except FileNotFoundError:
        pass

    for file in os.listdir(source):
        if file.endswith('zip'):

            os.mkdir('./temp_folder/')
            shutil.move(file, './temp_folder/')
            shutil.unpack_archive(f'./temp_folder/{file}', "./temp_folder/", "zip")
            os.remove(f"./temp_folder/{file}")
            unziped_files = os.listdir('./temp_folder/')

            for folder in unziped_files:
                if os.path.isdir(f'./temp_folder/{folder}'):
                    if folder != '__MACOSX':
                        folder_files = os.listdir(f'./temp_folder/{folder}')

                        for file in folder_files:
                            shutil.move(f'./temp_folder/{folder}/{file}', destination)
                else:
                    shutil.move(f'./temp_folder/{folder}', destination)
            shutil.rmtree('./temp_folder/')

        elif file.endswith('wav') or file.endswith('mp3'):
          try:
            shutil.move(source + file, destination)
          except:
            pass
