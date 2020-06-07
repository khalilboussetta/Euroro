# Import standard library
import random
from math import sqrt
from typing import Tuple

# Import modules
import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image
import glob

cred_dict = {'Guest':0,'Registered':1,'Advanced':2}

initial = True

def main():
    # global image_path, image_name
    # # Sidebar
    # st.sidebar.header("Parameters")
    # st.sidebar.markdown("User authorisation")
    # user = st.sidebar.radio(
    #     "User Type", options=['User1', 'User2', 'User3'], index=0
    #     )

    # #n_sprites = 1
    # # Main Page
    # st.title("Labeling Tool for nonwovens")
    # st.markdown(
    #     """
    #     ## Instructions
    #     Rate the following image for the presence of nonwovens
        
    #     Press next image once you have rated this image to pass to the next one
    # """
    # )

    # slot = st.empty()

    # st.markdown("**Rate this image**")

    # rating = st.radio(
    #     "", options=[0, 1, 2, 3], index=0
    # )

    # if st.button("Next"):
    #     file = open("last_img.txt", "r")
    #     image_name = file.readline()[:-1]
    #     image_path = file.readline()[:-1]
    #     file.close()

    #     save_rating(rating, user, image_name)
    #     with st.spinner("Loading image..."):
    #         image_name, image_path = load_image(
    #             user=user
    #         )

    #         if image_path is not None:
    #             file = open("last_img.txt", "w")
    #             file.write(image_name + '\n')
    #             file.write(image_path + '\n')
    #             file.close()

    #             fig = Image.open(image_path)
    #             slot.image(fig, use_column_width = True)
    #         else:
    #             slot.markdown("**You have processed all available images. Thank you for helping**")
    # else:
    #     with st.spinner("Loading image..."):
    #         # image_name, image_path = load_image(
    #         #     user=user
    #         # )
    #         file = open("last_img.txt", "r")
    #         image_name = file.readline()[:-1]
    #         image_path = file.readline()[:-1]
    #         file.close()
    #         fig = Image.open(image_path)
    #         slot.image(fig, use_column_width=True)



    # #st.pyplot(fig=fig, bbox_inches="tight")

    # st.markdown("**Debug Section**")

    # st.dataframe(user_df)
    # st.empty()
    # st.dataframe(img_df)
    # st.markdown(
    #     """
    #     ---
    #     This application is made with :heart:
         
    #     Lutz chyetnek fi rasou
    # """
    #  )
    file = open("last_img.txt", "r")
    image_name = file.readline()[:-1]
    image_path = file.readline()[:-1]
    file.close()
    st.title(image_name)
    st.title(image_path)
    

  


def load_image(
    user: str
):
    """Main function for creating sprites
    Parameters
    ----------
    n_sprites : int
        Number of sprites to generate
    n_iters : int
        Number of iterations to run the simulator
    repro_rate : int
        Inverse reproduction rate
    stasis_rate : int
        Stasis rate
    """
    logger.info("Loading Image")


    # # Generate plot based on the grid size
    # n_grid = int(sqrt(n_sprites))
    #
    # img= cv2.imread('Zimmer1.jpg')
    # fig = cv2.resize(img,(320,320))
    # fig, axs = plt.subplots(n_grid, n_grid, figsize=(3, 3))
    # #axs = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    # axs.imshow(img,  interpolation="nearest")
    lvl, rated_imgs = get_user_data(user)
    image_name, image_path = get_image_data(lvl, rated_imgs)
    return image_name, image_path

@st.cache(allow_output_mutation=True)
def generate_image_dataframe():
    img_list = glob.glob('**/*.jpg', recursive=True)
    df = pd.DataFrame(columns=['Name', 'Auth_lvl', 'Ratings', 'Path'])
    for img_path in img_list:
        name = img_path[5:-4]
        lvl = int(img_path[3])
        df.loc[-1] = [name, lvl, '{}', img_path]  # adding a row
        df.index = df.index + 1  # shifting index

    df = df.sort_index()  # sorting by index

    return df

@st.cache(allow_output_mutation=True)
def generate_user_dataframe(filename):
    user_df = pd.read_csv(filename, index_col=0)
    user_df[user_df.isnull()] = None
    return user_df


def get_user_data(user):
    user_data = user_df[user_df['Name'] == user]
    user_lvl = cred_dict[user_data['Type'].values[0]]
    user_rtg = user_data['Ratings'].values[0]
    if user_rtg == '{}':
        rated_imgs = []
    else:
        rated_imgs = list(user_rtg.keys())

    return user_lvl, rated_imgs

def get_image_data(lvl, rated_imgs):
    filtered_df = img_df[img_df['Auth_lvl']<=lvl]
    img_list = list(filtered_df['Name'].values)
    left_imgs = [i for i in img_list if i not in rated_imgs]
    if len(left_imgs)!=0:
        id = random.randint(0, len(left_imgs) - 1)
        image_name = left_imgs[id]
        image_path = filtered_df[filtered_df['Name']==image_name]['Path'].values[0]
        return image_name, image_path
    else:
        return None, None

def save_rating(rating, user, image):

    # Update user dataframe
    user_idx = user_df[user_df['Name'] == user].index.values[0]
    user_ratings = user_df.loc[user_idx, 'Ratings']

    if user_ratings == '{}':
        user_df.at[user_idx,'Ratings'] = {image:rating}
    else:
        user_ratings[image]=rating
        user_df.at[user_idx, 'Ratings'] = user_ratings


    # Update images dataframe
    img_idx = img_df[img_df['Name'] == image].index.values[0]
    img_ratings = img_df.loc[img_idx, 'Ratings']

    if img_ratings == '{}':
        img_df.at[img_idx, 'Ratings'] = {user: rating}
    else:
        img_ratings[user] = rating
        img_df.at[img_idx, 'Ratings'] = img_ratings



# Users Authorizations
user_df = generate_user_dataframe('Users.csv')
# Image DataFrame
img_df = generate_image_dataframe()



main()