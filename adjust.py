from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import matplotlib.pyplot as plt

# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = True
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)



colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
 #   print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
  #      print("".join(line))
        
        
################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations


     
################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION


last_mario_x = None
last_mario_y = None
is_in_air = False
static_frame_count = 0
last_enemy_x = None
last_action = None
x_positions = [0] * 10
stair_locator = 0
pipe_stuck = 0
MARIO_MAX_JUMP_HEIGHT = 64
GROUND_LEVEL_Y = 194
stair_x_threshold = 20
stair_y_threshold = 10
jump_counter = 0
is_short_jumping = False
short_jump_threshhold = 2
long_jump_threshold = 50
is_long_jumping = False
long_jump_counter = 0

momentum = 0

def make_action(screen, info, step, env, prev_action):

    global last_mario_x, static_frame_count, last_enemy_x, last_mario_y, is_in_air, last_action, stair_locator, pipe_stuck, MARIO_MAX_JUMP_HEIGHT, GROUND_LEVEL_Y, stair_x_threshold, stair_y_threshold, jump_counter, is_short_jumping, short_jump_threshhold, momentum, long_jump_threshold, is_long_jumping, long_jump_counter


    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
   # if PRINT_GRID and step % 100 == 0:
       # print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        #print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
    
    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
           # print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
         #   print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")

        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
           # print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
       # print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            #print("Mario's location on screen:",
            #      mario_x, mario_y, f"({object_name} mario)")
        
        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        #print("Mario's location in world:",
              #mario_world_x, mario_world_y, f"({mario_status} mario)")

    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press 'right' button
    #              action = 2 means press 'right' and 'A' buttons at the same time




    #Mario Jump path checker
    def is_overlapping(top_left1, bottom_right1, top_left2, dimensions2):
        overlap_x = top_left2[0] + dimensions2[0] > top_left1[0] and top_left2[0] < bottom_right1[0]
        overlap_y = top_left2[1] + dimensions2[1] > top_left1[1] and top_left2[1] < bottom_right1[1]
        return overlap_x and overlap_y

    def gap_check(mario_location, block_locations):
        global long_jump_threshold, is_long_jumping, momentum
        mario_x, mario_y = mario_location
        scan_width = 30

        for x in range (mario_x, mario_x + scan_width):
            block_below = False
            for block in block_locations:
                block_x , block_y = block[0]
                block_width, block_height = block[1]

                #check if block horizontal overlap with mario
                if block_x <= x < block_x + block_width:
                    #check if block is directly below mario and above ground_level
                    if block_y > mario_y and block_y >= GROUND_LEVEL_Y:
                        block_below = True
                       # print("Block do be below here at ", block_x," ", block_y)
                        break

            if not block_below:
               # print(f"Gap is detected at x range : {mario_x} to {x}, y range : {mario_y} to {GROUND_LEVEL_Y}")
                
                distance_to_gap = mario_x - x

                if 0 < distance_to_gap < 10:
                   # print ("the running distance is :, therfore go left", distance_to_gap)
                    momentum = abs(momentum - 5)
                    return 6 # move left for running start   
                if distance_to_gap > 15:
                    print("getting a run start")
                    return 3 #i gotta run up
                if distance_to_gap < 10:

                    if not is_long_jumping:
                        is_long_jumping = True
                        long_jump_counter = 0
                        return 4 #start jumping
                    elif is_long_jumping:
                        long_jump_counter += 1 
                        print("long jump counter is ", long_jump_counter)
                        if long_jump_counter <= long_jump_threshold:
                            return 4
                        else:
                            is_long_jumping = False #reset

                        

                    print("Jumpin due to being close enouf")
                    return 4 # should be able to jump now
                

    def detect_pipe(mario_location, block_locations):
        mario_x, mario_y = mario_location
        distance_threshold = 60

        for block in block_locations:
            block_x, block_y = block[0]
            block_name = block[2]

           # print(f"There is a block with name {block_name} at x: {block_x}, y: {block_y}")

            #check for pipe infront
            if "pipe" in block_name and block_x > mario_x:

                
                if(mario_y < block_y):
                  #  print("Mario is above pipe")
                    return 1
                
                if (block_x - mario_x < 20):
                    #print("Too close to pipe")
                    return 1
                if (block_x - mario_x) < distance_threshold:
                   # print("Pipe is within mario jump threshold")
                    return 3

        return False


    def stair_located(mario_location, block_locations):
        #Function to decide what to do when met with stairs
        mario_x , mario_y = mario_location

        for block in block_locations:
            block_x, block_y = block[0]
            block_name = block[2]

            #check if block infront
           # print("Mario_x position is ", mario_x, mario_y)
           # print("The block x and y position are ", block_x, block_y)
            if block_name == "block" and (block_x - mario_x) <= stair_x_threshold and abs(mario_y - block_y) <= stair_y_threshold:
             #   print("There is a stair infront")
                return True
        return False


    
    

    if mario_locations:
        location, dimensions, object_name = mario_locations[0]
        mario_x, mario_y = location
        mario_width, mario_height = dimensions
        #print("Mario Location on screen and dimensisions:",
              #mario_x, mario_y, mario_width, mario_height, f"({object_name} mario)")
        


        #Jump_path
        jump_path_top_left = (mario_x, mario_y - MARIO_MAX_JUMP_HEIGHT)
        jump_path_bottom_right = (mario_x + mario_width, mario_y)

    

        if last_mario_x is None:
            last_mario_x = mario_x
           # print("Last_mario_x updated to ", last_mario_x)

        if mario_x == last_mario_x:
            static_frame_count +=1
           # print("static frame count updated to : ",static_frame_count)
           # print("mario x location is ", mario_x, last_mario_x)
        
        else:
            last_mario_x = mario_x
            static_frame_count = 0
        
        if static_frame_count >= 100:
           # print("This guy stuck for longer than 100 frames")
            static_frame_count = 0 # reset count

            if last_action == 4:
                last_action = 0
               # print("Letting go of jump")
                momentum -= 1
                return 0
                
            else:
                last_action = 4
               # print("Jumping cuz stuck")
                return 4 # changed to let go of jump button
        

        if last_mario_y is not None and mario_y > last_mario_y:
            is_in_air = True
            #print("MARIO in AIR")

        if last_mario_y is not None and mario_y < last_mario_y and is_in_air:
            is_in_air = False
            #print("Jump was released")
            #return 0  Release the jump button
        #print(f"Last_mario_y:{last_mario_y}, mario_y: {mario_y} ")
        last_mario_y = mario_y
        #print(f"Last_mario_y:{last_mario_y}, mario_y: {mario_y} ")

        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            #print("enemy:", x, y, width, height, enemy_name)
            #print(f"Mario x position is {mario_x} and y position is {mario_y}")

            if (momentum >= 45):
                momentum = 45
            
            if (momentum <= -40):
                momentum = -40

            if(y <= GROUND_LEVEL_Y):

                enemy_distance = abs(mario_x - x)

               # print(f"the momentum is{momentum}")


                if (55 <= enemy_distance <= 60):
                    if (momentum <= 15):
                            short_jump_threshhold = 11
                    if (momentum <= 40):
                                short_jump_threshhold = 10
                    if (momentum > 40):
                                short_jump_threshhold = 2



                if (enemy_distance <= 52):
                    print("The distance of enemy is :", enemy_distance)
                    print("Enemy is in jumpable range")

                    block_in_path = False
                    while True:
                        for block in block_locations:
                            block_location, block_dimensions, block_name = block

                            if is_overlapping(jump_path_top_left, jump_path_bottom_right, block_location, block_dimensions):
                                block_in_path = True
                                break
                        if block_in_path:
                            print("Block was in the way")
                            momentum -= 1
                            return 6 # Move left
                        else:
                            print("Block not in jump path and i should jump now")
                            

                            min_air_distance_enemy = y + 5
                            min_air_distance_mario = x + 5

                            if min_air_distance_enemy < mario_y : #enemy in the air above mario
                                
                                
                                if (mario_x < x):
                                    print("Enemy above on the right move to left")
                                    momentum -= 1
                                    return 6 # move left to avoid
                                if (mario_x > x):   
                                    print("Enemy above on the left move to right")
                                    return 3 #move right to above
                                    
                            if is_in_air ==  True and min_air_distance_mario < y:
                                if (mario_x < x):
                                    print("Mario needs to move right to jump on him")
                                    momentum += 1
                                    return 1
                                
                                if (mario_x > x):
                                    print("Mario needs to move left to jump on him")
                                    momentum -=1
                                    return 6
                                if (mario_x == x):
                                    print("Mario is directly on top")
                                    momentum -= 1
                                    return 0
                                
                            
                            
                            
                            if (enemy_distance < 25 and x - mario_x > 0):
                                print("ENEMY too close gotta move left")
                                return 6# move left away from enemy as too close to jump
                            
                            if min_air_distance_enemy < mario_y : #enemy in the air above mario
                                
                                
                                if (mario_x < x):
                                    print("Enemy above on the right move to left")
                                    momentum = abs(momentum - 5)
                                    return 6 # move left to avoid
                                if (mario_x > x):   
                                    print("Enemy above on the left move to right")
                                    return 3 #move right to above


                            if not is_short_jumping:
                                is_short_jumping = True
                                jump_counter = 0
                                print("Started short jump")
                                return 4 # start jump
                            elif is_short_jumping:
                                print(f"This is the threshold doing it with {short_jump_threshhold}")
                                print("Doing the short jump")
                                jump_counter += 1
                                if jump_counter < short_jump_threshhold:
                                    return 4
                                else:
                                    print("Short jump reset time")
                                    is_short_jumping = False #reset jump state
                                    momentum -= 0
                                    return 0

                

        
        
        
        
        
        
        if mario_x < 10:
            return 3 # Run right
        
        
        
        
        
        
        
        
        if gap_check(location, block_locations) == 6:
            return 6
        if gap_check(location, block_locations) == 3:
            return 3
        if gap_check(location, block_locations) == 4:
          #  print("Jumping due to the gap")
            return 4
        
        #pipe method use

       

        pipe_check = detect_pipe(location,block_locations)

        

        if  pipe_check != False:
            if pipe_check == 1:
             #   print("printing jumpin due to the pipe check")
               # print("pipe do be infront")
                pipe_stuck += 1
                if pipe_stuck >= 100:
                  #  print("Probs stuck in between pipe, letting go of jump")
                    pipe_stuck = 0 #rest
                    momentum -= 1
                    return 0
                return 4
            if pipe_check == 2:
               # print("I should be able to jump due to pipe")
                return 4
            if pipe_check == 3:
                return 3


        if stair_located(location, block_locations):
          #  print("Stair do be infront")
            stair_locator += 1
            if stair_locator >= 100:
             #   print("Probs stuck in between hole, letting go of jump")
                stair_locator = 0 #rest
                return 0
            return 4

    
                      
                    
    """ Potential Questionblock method for more score
    for block in block_locations:
            block_location, block_dimensions, block_name = block
            block_x, block_y = block_location

            if (block_name == "question_block"):
                print("This is the question block coords : " ,block_x, block_y)
                if (block_x == mario_x):
                    print("They have intersected")
               # print("Mario is directly under block, the coords for mario and block are : ", mario_x, block_x)
                return 4
    """
    

            



        
    
    #print("I reached i should be moving right")
    momentum += 1
    return 1


    if step % 10 == 0:
        # I have no strategy at the moment, so I'll choose a random action.
        action = env.action_space.sample()
        return action


    else:
        # With a random agent, I found that choosing the same random action
        # 10 times in a row leads to slightly better performance than choosing
        # a new random action every step.
        return prev_action

################################################################################


env = gym.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
actions_per_episode = []
scores_per_episode = []
distances_per_episode = []
coins_per_episode = []
in_game_times = []
obs = None
done = True
env.reset()
action_count = 0
episode_count = 0
max_episodes = 10 

for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    action_count += 1

    if info.get('flag_get'):
            print("Mario reached the flag!")
            time_taken_game = 400 - info['time']
            in_game_times.append(time_taken_game)
            print(f"Time taken in game: {time_taken_game}")

    done = terminated or truncated
    if done:
        episode_count += 1
        print(f"Episode {episode_count} finished after {action_count} actions")
        
        actions_per_episode.append(action_count)
        scores_per_episode.append(info['score'])
        distances_per_episode.append(info['x_pos'])
        coins_per_episode.append(info['coins'])

        action_count = 0        
        env.reset()

        if episode_count >= max_episodes:
            break
env.close()


# Plotting the metrics
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(actions_per_episode, label='Actions Taken')
plt.xlabel('Episode')
plt.ylabel('Actions Taken')
plt.title('Actions Taken per Episode')

plt.subplot(2, 3, 2)
plt.plot(in_game_times, label='In-Game Time', color='orange')
plt.xlabel('Episode of Reaching Flag')
plt.ylabel('Time (in-game time)')
plt.title('In-Game Time to Reach Flag')


plt.subplot(2, 3, 3)
plt.plot(scores_per_episode, label='Episode Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title(' Reward per Episode')

plt.subplot(2, 3, 4)
plt.plot(distances_per_episode, label='Distance Reached', color='purple')
plt.xlabel('Episode')
plt.ylabel('Distance Reached')
plt.title('Distance Reached per Episode')

plt.subplot(2, 3, 5)
plt.plot(coins_per_episode, label='Coins Collected', color='gold')
plt.xlabel('Episode')
plt.ylabel('Coins Collected')
plt.title('Coins Collected per Episode')

plt.tight_layout()
plt.show()
