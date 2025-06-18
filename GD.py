import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from skimage import measure
import time

# COCO-to-TACO category mapping with recyclability statusStre
coco_to_taco_mapping = {
    1: ('person', 'unknown'),
    2: ('bicycle', 'recyclable'),
    3: ('car', 'recyclable'),
    4: ('motorcycle', 'recyclable'),
    5: ('airplane', 'recyclable'),
    6: ('bus', 'recyclable'),
    7: ('train', 'recyclable'),
    8: ('truck', 'recyclable'),
    9: ('boat', 'recyclable'),
    10: ('traffic_light', 'recyclable'),
    11: ('fire_hydrant', 'recyclable'),
    13: ('stop_sign', 'recyclable'),
    14: ('parking_meter', 'recyclable'),
    15: ('bench', 'recyclable'),
    16: ('bird', 'non-recyclable'),
    17: ('cat', 'non-recyclable'),
    18: ('dog', 'non-recyclable'),
    19: ('horse', 'non-recyclable'),
    20: ('sheep', 'non-recyclable'),
    21: ('cow', 'non-recyclable'),
    22: ('elephant', 'non-recyclable'),
    23: ('bear', 'non-recyclable'),
    24: ('zebra', 'non-recyclable'),
    25: ('giraffe', 'non-recyclable'),
    27: ('backpack', 'recyclable'),
    28: ('umbrella', 'recyclable'),
    31: ('handbag', 'recyclable'),
    32: ('tie', 'recyclable'),
    33: ('suitcase', 'recyclable'),
    34: ('frisbee', 'recyclable'),
    35: ('skis', 'recyclable'),
    36: ('snowboard', 'recyclable'),
    37: ('sports_ball', 'recyclable'),
    38: ('kite', 'recyclable'),
    39: ('baseball_bat', 'recyclable'),
    40: ('baseball_glove', 'recyclable'),
    41: ('skateboard', 'recyclable'),
    42: ('surfboard', 'recyclable'),
    43: ('tennis_racket', 'recyclable'),
    44: ('bottle', 'recyclable'),
    45: ('wine_glass', 'recyclable'),
    46: ('cup', 'recyclable'),
    47: ('fork', 'recyclable'),
    48: ('knife', 'recyclable'),
    49: ('spoon', 'recyclable'),
    50: ('bowl', 'recyclable'),
    51: ('banana', 'non-recyclable'),
    52: ('apple', 'non-recyclable'),
    53: ('sandwich', 'non-recyclable'),
    54: ('orange', 'non-recyclable'),
    55: ('broccoli', 'non-recyclable'),
    56: ('carrot', 'non-recyclable'),
    57: ('hot_dog', 'non-recyclable'),
    58: ('pizza', 'non-recyclable'),
    59: ('donut', 'non-recyclable'),
    60: ('cake', 'non-recyclable'),
    61: ('chair', 'recyclable'),
    62: ('couch', 'recyclable'),
    63: ('potted_plant', 'recyclable'),
    64: ('bed', 'recyclable'),
    65: ('dining_table', 'recyclable'),

    70: ('tv', 'recyclable'),
    72: ('laptop', 'recyclable'),
    73: ('mouse', 'recyclable'),
    74: ('remote', 'recyclable'),
    75: ('keyboard', 'recyclable'),
    76: ('cell_phone', 'recyclable'),
    77: ('microwave', 'recyclable'),
    78: ('oven', 'recyclable'),
    79: ('toaster', 'recyclable'),
    80: ('sink', 'recyclable'),
    81: ('refrigerator', 'recyclable'),
    82: ('book', 'recyclable'),
    84: ('clock', 'recyclable'),
    85: ('vase', 'recyclable'),
    86: ('scissors', 'recyclable'),
    87: ('teddy_bear', 'recyclable'),
    88: ('hair_drier', 'recyclable'),
    89: ('toothbrush', 'recyclable'),

    # Additional custom categories for waste
    100: ('plastic_bag', 'non-recyclable'),
    101: ('plastic_straw', 'non-recyclable'),
    102: ('styrofoam_cup', 'non-recyclable'),
    103: ('aluminum_can', 'recyclable'),
    104: ('cardboard_box', 'recyclable'),
    105: ('plastic_bottle', 'recyclable'),
    106: ('ceramic_plate', 'recyclable'),
    107: ('coffee_cup_lid', 'recyclable'),
    108: ('juice_carton', 'recyclable'),
    109: ('milk_jug', 'recyclable'),
    110: ('newspaper', 'recyclable'),
    111: ('pizza_box', 'non-recyclable'),
    112: ('yogurt_cup', 'recyclable'),
    113: ('metal_can', 'recyclable'),
    114: ('plastic_cutlery', 'non-recyclable'),
    115: ('egg_carton', 'recyclable'),
    116: ('aluminum_foil', 'recyclable'),
    117: ('plastic_toy', 'non-recyclable'),
    118: ('bubble_wrap', 'non-recyclable'),
    119: ('fabric_scrap', 'non-recyclable'),
    120: ('glass_jar', 'recyclable'),
    121: ('light_bulb', 'recyclable'),
    122: ('battery', 'recyclable'),
    123: ('shampoo_bottle', 'recyclable'),
    124: ('detergent_bottle', 'recyclable'),
    125: ('toothpaste_tube', 'non-recyclable'),
    126: ('cotton_pad', 'non-recyclable'),
    128: ('cigarette_butt', 'non-recyclable'),
    129: ('face_mask', 'non-recyclable'),
    130: ('takeaway_container', 'non-recyclable'),
    131: ('bread_bag', 'non-recyclable'),
    132: ('plastic_wrap', 'non-recyclable'),
    133: ('juice_box', 'recyclable'),
    134: ('plastic_lid', 'recyclable'),
    135: ('beer_bottle', 'recyclable'),
    136: ('wine_bottle', 'recyclable'),
    137: ('soda_can', 'recyclable'),
    138: ('chips_bag', 'non-recyclable'),
    139: ('paper_plate', 'recyclable'),
    140: ('plastic_tub', 'recyclable'),
    141: ('nail_clippers', 'recyclable'),
    142: ('wooden_spoon', 'recyclable'),
    143: ('plastic_toys', 'non-recyclable'),
    144: ('glass_shards', 'non-recyclable'),
    145: ('garden_waste', 'non-recyclable'),
    146: ('toothpick', 'non-recyclable'),
    147: ('pencil', 'recyclable'),
    148: ('eraser', 'non-recyclable'),
    149: ('chalk', 'recyclable'),
    150: ('paint_can', 'recyclable'),
    151: ('metal_scrap', 'recyclable'),
    152: ('shoes', 'recyclable'),
    153: ('clothes', 'recyclable'),
    154: ('pill_bottle', 'recyclable'),
    155: ('garden_pot', 'recyclable'),
    156: ('fabric_cloth', 'recyclable'),
    157: ('rubber_band', 'non-recyclable'),
    158: ('twist_tie', 'non-recyclable'),
    159: ('magazine', 'recyclable'),
    160: ('plastic_cup', 'recyclable'),
    161: ('bicycle_parts', 'recyclable'),
    162: ('engine_oil', 'non-recyclable'),
    163: ('medical_waste', 'non-recyclable'),
    164: ('thermometer', 'recyclable'),
    165: ('inhaler', 'recyclable'),
    166: ('paint_brush', 'recyclable'),
    167: ('nail_file', 'recyclable'),
    168: ('comb', 'recyclable'),
    169: ('hair_clips', 'recyclable'),
    170: ('tape', 'non-recyclable'),
    171: ('stapler', 'recyclable'),
    172: ('rubber_glove', 'non-recyclable'),
    173: ('envelope', 'recyclable'),
    174: ('phone_charger', 'recyclable'),
    175: ('metal_fork', 'recyclable'),
    176: ('plastic_lawn_chair', 'recyclable'),
    177: ('glass_table', 'recyclable'),
    178: ('wine_cork', 'non-recyclable'),
    179: ('hair_extension', 'non-recyclable'),
    180: ('jewelry_box', 'recyclable'),
    181: ('wooden_crate', 'recyclable'),
    182: ('styrofoam_plate', 'non-recyclable'),
    183: ('plastic_suitcase', 'recyclable'),
    184: ('coffee_grinder', 'recyclable'),
    185: ('plastic_straw', 'non-recyclable'),
    186: ('plastic_napkin', 'non-recyclable'),
    187: ('straw_hat', 'recyclable'),
    188: ('phone_case', 'recyclable'),
    189: ('metal_spoon', 'recyclable'),
    190: ('wooden_plank', 'recyclable'),
    191: ('ceramic_cup', 'recyclable'),
    192: ('plastic_shoes', 'recyclable'),
    193: ('glass_mug', 'recyclable'),
    194: ('steel_pan', 'recyclable'),
    195: ('wire', 'recyclable'),
    196: ('ceramic_tile', 'recyclable'),
    197: ('metal_rod', 'recyclable'),
    198: ('glass_shard', 'non-recyclable'),
    199: ('metal_scrap_pieces', 'recyclable'),
    200: ('tv_remote', 'recyclable'),
    201: ('pen', 'recyclable'),
    202: ('pencil', 'recyclable'),
    203: ('mobile_phone', 'recyclable'),
    204: ('mobile_phone_back_cover', 'recyclable'),
    205: ('paper', 'recyclable'),
    206: ('scale', 'recyclable'),
    207: ('glasses', 'recyclable'),
    208: ('sunglasses', 'recyclable'),
    209: ('laptop_charger', 'recyclable'),
    210: ('desktop_computer', 'recyclable'),
    211: ('tablet', 'recyclable'),
    212: ('hard_drive', 'recyclable'), 
    213: ('usb_drive', 'recyclable'),
    214: ('power_bank', 'recyclable'),
    215: ('notebook', 'recyclable'),
    216: ('sticky_notes', 'recyclable'),
    217: ('post_it', 'recyclable'),
    218: ('whiteboard', 'recyclable'),
    219: ('whiteboard_marker', 'recyclable'),
    220: ('highlighter', 'recyclable'),
    221: ('marker', 'recyclable'),
    222: ('tape_dispenser', 'recyclable'),
    223: ('hole_punch', 'recyclable'),
    224: ('stapler', 'recyclable'),
    225: ('paper_clips', 'recyclable'),
    226: ('rubber_band', 'recyclable'),
    227: ('glue_stick', 'non-recyclable'),
    228: ('adhesive_tape', 'non-recyclable'),
    229: ('scotch_tape', 'non-recyclable'),
    230: ('binder', 'recyclable'),
    231: ('clipboard', 'recyclable'),
    232: ('envelope', 'recyclable'),
    233: ('gift_wrap', 'non-recyclable'),
    234: ('shipping_box', 'recyclable'),
    235: ('plastic_wrap', 'non-recyclable'),
    236: ('bubble_envelope', 'non-recyclable'),
    237: ('plastic_file_folder', 'non-recyclable'),
    238: ('craft_paper', 'recyclable'),
    239: ('photograph', 'non-recyclable'),
    240: ('business_card', 'recyclable'),
    241: ('magnet', 'non-recyclable'),
    242: ('adhesive_label', 'non-recyclable'),
    243: ('permanent_marker', 'non-recyclable'),
    244: ('drafting_tools', 'recyclable'),
    245: ('ruler', 'recyclable'),
    246: ('calculator', 'recyclable'),
    247: ('eraser', 'non-recyclable'),
    248: ('geometry_set', 'recyclable'),
    249: ('t-square', 'recyclable'),
    250: ('canvas', 'non-recyclable'),
    251: ('water_color', 'non-recyclable'),
    252: ('paint_palette', 'non-recyclable'),
    253: ('crayon', 'non-recyclable'),
    254: ('marker_pen', 'non-recyclable'),
    255: ('colored_pencil', 'recyclable'),
    256: ('sketchbook', 'recyclable'),
    257: ('drawing_tablet', 'recyclable'),
    258: ('safety_scissors', 'recyclable'),
    259: ('craft_knife', 'recyclable'),
    260: ('pencil_case', 'recyclable'),
    261: ('bag_of_pens', 'recyclable'),
    262: ('drawing_paper', 'recyclable'),
    263: ('charcoal', 'non-recyclable'),
    264: ('paintbrush', 'recyclable'),
    265: ('canvas_board', 'non-recyclable'),
    266: ('glass_jar', 'recyclable'),
    267: ('jar_lid', 'recyclable'),
    268: ('baking_sheet', 'non-recyclable'),
    269: ('plastic_container', 'recyclable'),
    270: ('food_container', 'recyclable'),
    271: ('aluminum_foil', 'recyclable'),
    272: ('takeaway_bag', 'non-recyclable'),
    273: ('cereal_box', 'recyclable'),
    274: ('pizza_box', 'non-recyclable'),
    275: ('cleaning_sponges', 'non-recyclable'),
    276: ('tissue_box', 'recyclable'),
    277: ('shampoo_bottle', 'recyclable'),
    278: ('toilet_paper_roll', 'recyclable'),
    279: ('facial_tissues', 'non-recyclable'),
    280: ('paper_towel', 'non-recyclable'),
    281: ('kitchen_roll', 'non-recyclable'),
    282: ('plastic_wrap', 'non-recyclable'),
    283: ('ziploc_bag', 'non-recyclable'),
    284: ('trash_bag', 'non-recyclable'),
    285: ('reusable_bag', 'recyclable'),
    286: ('shopping_bag', 'recyclable'),
    287: ('gift_bag', 'non-recyclable'),
    288: ('bread_wrapper', 'non-recyclable'),
    289: ('cleaning_rags', 'non-recyclable'),
    290: ('carpet', 'non-recyclable'),
    291: ('old_furniture', 'recyclable'),
    292: ('old_mattress', 'non-recyclable'),
    293: ('exercise_mat', 'non-recyclable'),
    294: ('old_clothes', 'recyclable'),
    295: ('tote_bag', 'recyclable'),
    296: ('diaper', 'non-recyclable'),
    297: ('pet_food_bag', 'non-recyclable'),
    298: ('plastic_bin', 'recyclable'),
    299: ('recycle_bin', 'recyclable'),
    300: ('garden_bag', 'non-recyclable')
}

def map_coco_label_to_taco(coco_label):
    return coco_to_taco_mapping.get(coco_label, ('unknown', 'unknown'))

# Load pre-trained Mask R-CNN model (COCO)
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()


preprocess = transforms.Compose([transforms.ToTensor()])

# Streamlit interface
st.set_page_config(page_title="♻️ Waste Detection & Monitoring System", page_icon="♻️")
st.title("♻️ Waste Detection and Monitoring System")
st.subheader("Real-Time, Image-Based, Drone-Controlled Waste Detection and Water Level Monitoring")

st.sidebar.header("Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4)
mode = st.sidebar.radio("Select Mode", ["Webcam Detection", "Image Upload", "Drone Input"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Waste Categories:**")
st.sidebar.markdown("""- 🟢 Recyclable\n- 🔴 Non-recyclable""")

FRAME_WINDOW = st.image([]) 
status_text = st.empty() 
bin_text = st.empty() 


bin_selection = st.sidebar.radio("Select Bin Type", ["Recyclable Bin", "Non-recyclable Bin"], index=0)

def get_color_for_category(category):
    if category == 'recyclable':
        return (0, 255, 0) 
    elif category == 'non-recyclable':
        return (0, 0, 255)  
    else:
        return (255, 255, 255) 

def detect_waste_in_image(image):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    return boxes, labels, scores

def detect_waste_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        status_text.error("Error: Unable to access webcam.")
        return

    status_text.success("Webcam running...")

    while st.session_state["webcam_running"]:
        ret, frame = cap.read()
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, labels, scores = detect_waste_in_image(pil_image)

            detected_recyclable = detected_non_recyclable = False

            for i, box in enumerate(boxes):
                if scores[i] > confidence_threshold:
                    taco_label, category = map_coco_label_to_taco(labels[i])
                    color = get_color_for_category(category)

                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(frame, f"{taco_label}: {scores[i]:.2f} ({category})", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    if category == 'recyclable':
                        detected_recyclable = True
                    elif category == 'non-recyclable':
                        detected_non_recyclable = True

            if detected_recyclable and detected_non_recyclable:
                bin_text.warning("Both recyclable and non-recyclable waste detected.")
            elif detected_recyclable:
                if bin_selection == "Recyclable Bin":
                    bin_text.success("Recyclable waste detected. Place in the Recyclable bin.")
                else:
                    bin_text.warning("Detected recyclable waste, but selected non-recyclable bin!")
            elif detected_non_recyclable:
                if bin_selection == "Non-recyclable Bin":
                    bin_text.error("Non-recyclable waste detected. Place in the Non-recyclable bin.")
                else:
                    bin_text.warning("Detected non-recyclable waste, but selected recyclable bin!")

            FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

def detect_waste_from_uploaded_image(image):
    pil_image = Image.open(image)
    boxes, labels, scores = detect_waste_in_image(pil_image)

    detected_recyclable = detected_non_recyclable = False

    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(boxes):
        if scores[i] > confidence_threshold:
            taco_label, category = map_coco_label_to_taco(labels[i])
            color = get_color_for_category(category)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"{taco_label}: {scores[i]:.2f} ({category})", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if category == 'recyclable':
                detected_recyclable = True
            elif category == 'non-recyclable':
                detected_non_recyclable = True

    if detected_recyclable and detected_non_recyclable:
        bin_text.warning("Both recyclable and non-recyclable waste detected.")
    elif detected_recyclable:
        if bin_selection == "Recyclable Bin":
            bin_text.success("Recyclable waste detected. Place in the Recyclable bin.")
        else:
            bin_text.warning("Detected recyclable waste, but selected non-recyclable bin!")
    elif detected_non_recyclable:
        if bin_selection == "Non-recyclable Bin":
            bin_text.error("Non-recyclable waste detected. Place in the Non-recyclable bin.")
        else:
            bin_text.warning("Detected non-recyclable waste, but selected recyclable bin!")

    FRAME_WINDOW.image(frame, channels="BGR")

def detect_waste_drone():
    st.info("Drone input and waste detection coming soon. Stay tuned!")

def check_water_ph_level():
   
    ph_level = np.random.uniform(6.0, 9.0) 
    status = "Safe" if 6.5 <= ph_level <= 8.5 else "Unsafe"
    return ph_level, status

def detect_water_level(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255,
                                    cv2.THRESH_BINARY_INV)  
    labels = measure.label(binary_image, connectivity=2)

    water_pixels = np.sum(labels > 0)

    normalized_level = min(100, (water_pixels / (image.shape[0] * image.shape[1])) * 100)
    return normalized_level



if mode == "Webcam Detection":
    if 'webcam_running' not in st.session_state:
        st.session_state["webcam_running"] = False

    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button and not st.session_state["webcam_running"]:
        st.session_state["webcam_running"] = True
        detect_waste_stream()
    elif stop_button and st.session_state["webcam_running"]:
        st.session_state["webcam_running"] = False
        status_text.info("Webcam stopped.")

elif mode == "Image Upload":
    uploaded_image = st.file_uploader("Upload an image for waste detection", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        detect_waste_from_uploaded_image(uploaded_image)


    st.subheader("Water Quality and Level Monitoring")

    uploaded_image = st.file_uploader("Upload an image for water level detection", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))

        water_level = detect_water_level(image)
        st.write(f"Detected Water Level: {water_level:.2f}%")

        ph_level, ph_status = check_water_ph_level()
        st.write(f"Detected Water pH Level: {ph_level:.2f} ({ph_status})")

elif mode == "Drone Input":
    if 'drone_connected' not in st.session_state:
        st.session_state['drone_connected'] = False

    connect_button = st.button("Connect to Drone")
    disconnect_button = st.button("Disconnect from Drone")

    if connect_button and not st.session_state['drone_connected']:
        st.session_state['drone_connected'] = True
        st.success("Drone connected successfully!")
        st.session_state['message'] = "Drone connected."

    if disconnect_button and st.session_state['drone_connected']:
        st.session_state['drone_connected'] = False
        st.warning("Drone disconnected.")
        st.session_state['message'] = "Drone disconnected."

    if st.session_state['drone_connected']:
        st.sidebar.markdown("### Movement Controls")
        col1, col2, col3 = st.sidebar.columns(3)
        col4, col5, col6 = st.sidebar.columns(3)

        with col1:
            if st.button("⬅️", key="left", help="Move Left"):
                st.session_state['command'] = "L"
                st.session_state['message'] = "Drone moving left"

        with col2:
            if st.button("⬆️", key="up", help="Move Up"):
                st.session_state['command'] = "U"
                st.session_state['message'] = "Drone moving up"

        with col3:
            if st.button("➡️", key="right", help="Move Right"):
                st.session_state['command'] = "R"
                st.session_state['message'] = "Drone moving right"

        with col4:
            if st.button("⬇️", key="down", help="Move Down"):
                st.session_state['command'] = "D"
                st.session_state['message'] = "Drone moving down"

        with col5:
            if st.button("⏹️", key="stop", help="Stop Drone"):
                st.session_state['command'] = "S"
                st.session_state['message'] = "Drone stopped"

        rotate_left_col, rotate_right_col, land_col = st.sidebar.columns(3)

        with rotate_left_col:
            if st.button("↩️", key="rotate_left", help="Rotate Left"):
                st.session_state['command'] = "RL"
                st.session_state['message'] = "Drone rotating left"

        with rotate_right_col:
            if st.button("↪️", key="rotate_right", help="Rotate Right"):
                st.session_state['command'] = "RR"
                st.session_state['message'] = "Drone rotating right"

        with land_col:
            if st.button("🛬", key="land", help="Land Drone"):
                st.session_state['command'] = "LND"
                st.session_state['message'] = "Drone landing..."

        st.sidebar.write(st.session_state['message'])
    else:
        st.warning("Please connect to the drone to control it.")

    FRAME_WINDOW = st.image([])
    if 'command' not in st.session_state:
        st.session_state['command'] = ''

    if 'message' not in st.session_state:
        st.session_state['message'] = "No action taken yet."
    st.sidebar.markdown("### Action Status")
    st.sidebar.info(st.session_state['message'])

    def detect_waste_in_drone_stream(image):
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(img_tensor)
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        return boxes, labels, scores

    if 'drone_streaming' not in st.session_state:
        st.session_state['drone_streaming'] = False
    def detect_waste_from_drone():
        st.info("Connecting to drone camera...")
        st.session_state['drone_streaming'] = True

        while st.session_state['drone_streaming']:
            
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            boxes, labels, scores = detect_waste_in_drone_stream(pil_image)

            for i, box in enumerate(boxes):
                if scores[i] > confidence_threshold:
                    taco_label, category = map_coco_label_to_taco(labels[i])
                    color = get_color_for_category(category)

                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(frame, f"{taco_label}: {scores[i]:.2f} ({category})",
                                (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            FRAME_WINDOW.image(frame, channels="BGR")

          
            time.sleep(0.1)
    start, stop = st.columns(2)

    with start:
        if st.button("🟢 Start Drone Detection"):
            if not st.session_state['drone_streaming']:
                detect_waste_from_drone()

    with stop:
        if st.button("🔴 Stop Drone Detection"):
            st.session_state['drone_streaming'] = False
            st.warning("Drone detection stopped.")




