import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
import threading
from .compat import UNIFORM_COLOR

annotation_enum_cases = [('91', 'airplane', 'airplane;aeroplane;plane'), ('127', 'animal', 'animal;animate;being;beast;brute;creature;fauna'), ('93', 'apparel', 'apparel;wearing;apparel;dress;clothes'), ('79', 'arcade machine', 'arcade;machine'), ('31', 'armchair', 'armchair'), ('139', 'ashcan', 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin'), ('87', 'awning', 'awning;sunshade;sunblind'), ('116', 'bag', 'bag'), ('120', 'ball', 'ball'), ('96', 'bannister', 'bannister;banister;balustrade;balusters;handrail'), ('78', 'bar', 'bar'), ('112', 'barrel', 'barrel;cask'), ('41', 'base', 'base;pedestal;stand'), ('113', 'basket', 'basket;handbasket'), ('38', 'bathtub', 'bathtub;bathing;tub;bath;tub'), ('8', 'bed', 'bed'), ('70', 'bench', 'bench'), ('128', 'bicycle', 'bicycle;bike;wheel;cycle'), ('132', 'blanket', 'blanket;cover'), ('64', 'blind', 'blind;screen'), ('77', 'boat', 'boat'), ('63', 'book', 'bookcase'), ('63', 'bookcase', 'bookcase'), ('89', 'booth', 'booth;cubicle;stall;kiosk'), ('99', 'bottle', 'bottle'), ('42', 'box', 'box'), ('62', 'bridge', 'bridge;span'), ('100', 'buffet', 'buffet;counter;sideboard'), ('2', 'building', 'building;edifice'), ('145', 'bulletin board', 'bulletin;board;notice;board'), ('81', 'bus', 'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle'), ('11', 'cabinet', 'cabinet'), ('107', 'canopy', 'canopy'), ('21', 'car', 'car;auto;automobile;machine;motorcar'), ('56', 'case', 'case;display;case;showcase;vitrine'), ('6', 'ceiling', 'ceiling'), ('20', 'chair', 'chair'), ('86', 'chandelier', 'chandelier;pendant;pendent'), ('45', 'chest of drawers', 'chest;of;drawers;chest;bureau;dresser'), ('149', 'clock', 'clock'), ('65', 'coffee table', 'coffee;table;cocktail;table'), ('43', 'column', 'column;pillar'), ('75', 'computer', 'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system'), ('106', 'conveyer belt', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter'), ('46', 'counter', 'counter'), ('71', 'countertop', 'countertop'), ('118', 'cradle', 'cradle'), ('142', 'crt screen', 'crt;screen'), ('19', 'curtain', 'curtain;drape;drapery;mantle;pall'), ('40', 'cushion', 'cushion'), ('34', 'desk', 'desk'), ('92', 'dirt track', 'dirt;track'), ('130', 'dishwasher', 'dishwasher;dish;washer;dishwashing;machine'), ('15', 'door', 'door;double;door'), ('14', 'earth', 'earth;ground'), ('97', 'escalator', 'escalator;moving;staircase;moving;stairway'), ('140', 'fan', 'fan'), ('33', 'fence', 'fence;fencing'), ('30', 'field', 'field'), ('50', 'fireplace', 'fireplace;hearth;open;fireplace'), ('150', 'flag', 'flag'), ('4', 'floor', 'floor;flooring'), ('67', 'flower', 'flower'), ('121', 'food', 'food;solid;food'), ('105', 'fountain', 'fountain'), ('148', 'glass', 'glass;drinking;glass'), ('52', 'grandstand', 'grandstand;covered;stand'), ('10', 'grass', 'grass'), ('69', 'hill', 'hill'), ('134', 'hood', 'hood;exhaust;hood'), ('26', 'house', 'house'), ('80', 'hovel', 'hovel;hut;hutch;shack;shanty'), ('74', 'land', 'kitchen;island'), ('74', 'kitchen island', 'kitchen;island'), ('129', 'lake', 'lake'), ('37', 'lamp', 'lamp'), ('83', 'light', 'light;light;source'), ('125', 'microwave', 'microwave;microwave;oven'), ('117', 'minibike', 'minibike;motorbike'), ('28', 'mirror', 'mirror'), ('144', 'monitor', 'monitor;monitoring;device'), ('17', 'mountain', 'mountain;mount'), ('98', 'ottoman', 'ottoman;pouf;pouffe;puff;hassock'), ('119', 'oven', 'oven'), ('23', 'painting', 'painting;picture'), ('73', 'palm', 'palm;palm;tree'), ('53', 'path', 'path'), ('13', 'person', 'person;individual;someone;somebody;mortal;soul'), ('141', 'pier', 'pier;wharf;wharfage;dock'), ('58', 'pillow', 'pillow'), ('18', 'plant', 'plant;flora;plant;life'), ('143', 'plate', 'plate'), ('109', 'plaything', 'plaything;toy'), ('94', 'pole', 'pole'), ('57', 'pool table', 'pool;table;billiard;table;snooker;table'), ('101', 'poster', 'poster;posting;placard;notice;bill;card'), ('147', 'radiator', 'radiator'), ('39', 'railing', 'railing;rail'), ('51', 'refrigerator', 'refrigerator;icebox'), ('61', 'river', 'river'), ('7', 'road', 'road;route'), ('35', 'rock', 'rock;stone'), ('29', 'rug', 'rug;carpet;carpeting'), ('55', 'runway', 'runway'), ('47', 'sand', 'sand'), ('135', 'sconce', 'sconce'), ('59', 'screen door', 'screen;door;screen'), ('59', 'screen', 'screen;door;screen'), ('133', 'sculpture', 'sculpture'), ('27', 'sea', 'sea'), ('32', 'seat', 'seat'), ('25', 'shelf', 'shelf'), ('104', 'ship', 'ship'), ('146', 'shower', 'shower'), ('12', 'sidewalk', 'sidewalk;pavement'), ('44', 'signboard', 'signboard;sign'), ('48', 'sink', 'sink'), ('3', 'sky', 'sky'), ('49', 'skyscraper', 'skyscraper'), ('24', 'sofa', 'sofa;couch;lounge'), ('102', 'stage', 'stage'), ('54', 'step', 'stairs;steps'), ('54', 'stairs', 'stairs;steps'), ('60', 'stairway', 'stairway;staircase'), ('72', 'stove', 'stove;kitchen;stove;range;kitchen;range;cooking;stove'), ('88', 'streetlight', 'streetlight;street;lamp'), ('110', 'swimming pool', 'swimming;pool;swimming;bath;natatorium'), ('76', 'swivel chair', 'swivel;chair'), ('16', 'table', 'table'), ('123', 'tank', 'tank;storage;tank'), ('90', 'television receiver', 'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box'), ('115', 'tent', 'tent;collapsible;shelter'), ('66', 'stool', 'toilet;can;commode;crapper;pot;potty;stool;throne'), ('66', 'pot', 'toilet;can;commode;crapper;pot;potty;stool;throne'), ('66', 'toilet', 'toilet;can;commode;crapper;pot;potty;stool;throne'), ('82', 'towel', 'towel'), ('85', 'tower', 'tower'), ('124', 'trade name', 'trade;name;brand;name;brand;marque'), ('137', 'traffic light', 'traffic;light;traffic;signal;stoplight'), ('138', 'tray', 'tray'), ('5', 'tree', 'tree'), ('84', 'truck', 'truck;motortruck'), ('103', 'van', 'van'), ('136', 'vase', 'vase'), ('1', 'wall', 'wall'), ('36', 'wardrobe', 'wardrobe;closet;press'), ('108', 'washer', 'washer;automatic;washer;washing;machine'), ('22', 'water', 'water'), ('114', 'waterfall', 'waterfall;falls'), ('9', 'windowpane', 'windowpane;window')]
annotation_colors = {'80': (1.0, 0.0, 0.996078431372549, 1.0), '140': (0.00392156862745098, 0.9607843137254902, 1.0, 1.0), '115': (0.43529411764705883, 0.8784313725490196, 0.996078431372549, 1.0), '33': (1.0, 0.7215686274509804, 0.023529411764705882, 1.0), '77': (0.6784313725490196, 1.0, 0.0, 1.0), '58': (0.0, 0.9215686274509803, 1.0, 1.0), '98': (0.996078431372549, 0.6, 0.0, 1.0), '74': (0.0, 1.0, 0.1607843137254902, 1.0), '150': (0.3568627450980392, 0.0, 0.996078431372549, 1.0), '117': (0.6431372549019608, 0.0, 1.0, 1.0), '18': (0.8, 1.0, 0.01568627450980392, 1.0), '145': (0.7215686274509804, 1.0, 0.0, 1.0), '63': (0.0, 1.0, 0.9607843137254902, 1.0), '9': (0.9019607843137255, 0.9019607843137255, 0.9019607843137255, 1.0), '34': (0.0392156862745098, 1.0, 0.2784313725490196, 1.0), '120': (1.0, 0.00392156862745098, 0.6352941176470588, 1.0), '61': (0.0392156862745098, 0.7843137254901961, 0.7843137254901961, 1.0), '32': (0.027450980392156862, 0.996078431372549, 0.8745098039215686, 1.0), '124': (0.5215686274509804, 0.996078431372549, 0.0, 1.0), '6': (0.4666666666666667, 0.47058823529411764, 0.3137254901960784, 1.0), '104': (1.0, 0.9215686274509803, 0.0, 1.0), '38': (0.4, 0.03137254901960784, 1.0, 1.0), '106': (0.5215686274509804, 0.0, 1.0, 1.0), '56': (0.00392156862745098, 0.0, 0.996078431372549, 1.0), '12': (0.9215686274509803, 1.0, 0.027450980392156862, 1.0), '19': (1.0, 0.2, 0.03137254901960784, 1.0), '70': (0.7607843137254902, 1.0, 0.00392156862745098, 1.0), '88': (0.00392156862745098, 0.2784313725490196, 1.0, 1.0), '62': (1.0, 0.3254901960784314, 0.00392156862745098, 1.0), '7': (0.5490196078431373, 0.5490196078431373, 0.5490196078431373, 1.0), '29': (1.0, 0.03529411764705882, 0.3607843137254902, 1.0), '82': (1.0, 0.0, 0.4, 1.0), '142': (0.47843137254901963, 0.00392156862745098, 1.0, 1.0), '11': (0.8784313725490196, 0.0196078431372549, 1.0, 1.0), '147': (1.0, 0.8392156862745098, 0.00784313725490196, 1.0), '76': (0.0392156862745098, 0.0, 1.0, 1.0), '146': (0.0, 0.5215686274509804, 0.996078431372549, 1.0), '54': (1.0, 0.8784313725490196, 0.00392156862745098, 1.0), '94': (0.2, 0.0, 1.0, 1.0), '59': (0.0, 0.8, 1.0, 1.0), '118': (0.6039215686274509, 0.0, 1.0, 1.0), '39': (1.0, 0.24313725490196078, 0.027450980392156862, 1.0), '101': (0.5607843137254902, 0.996078431372549, 0.00392156862745098, 1.0), '46': (0.9254901960784314, 0.043137254901960784, 1.0, 1.0), '110': (0.0, 0.7215686274509804, 1.0, 1.0), '45': (0.023529411764705882, 0.2, 1.0, 1.0), '93': (0.0, 0.44313725490196076, 0.996078431372549, 1.0), '13': (0.5882352941176471, 0.0196078431372549, 0.24313725490196078, 1.0), '50': (0.9803921568627451, 0.03529411764705882, 0.058823529411764705, 1.0), '64': (0.00392156862745098, 0.23921568627450981, 1.0, 1.0), '30': (0.43529411764705883, 0.03529411764705882, 1.0, 1.0), '99': (0.0, 1.0, 0.043137254901960784, 1.0), '35': (1.0, 0.1568627450980392, 0.03529411764705882, 1.0), '125': (0.996078431372549, 0.0, 0.9176470588235294, 1.0), '105': (0.03137254901960784, 0.7215686274509804, 0.6705882352941176, 1.0), '135': (0.0, 0.1607843137254902, 1.0, 1.0), '53': (1.0, 0.11764705882352941, 0.0, 1.0), '66': (0.0, 1.0, 0.5215686274509804, 1.0), '8': (0.8, 0.0196078431372549, 0.996078431372549, 1.0), '26': (1.0, 0.03137254901960784, 0.8666666666666667, 1.0), '14': (0.47058823529411764, 0.47058823529411764, 0.27450980392156865, 1.0), '73': (0.0, 0.3215686274509804, 0.996078431372549, 1.0), '23': (1.0, 0.023529411764705882, 0.2, 1.0), '24': (0.043137254901960784, 0.4, 1.0, 1.0), '97': (0.0, 1.0, 0.6392156862745098, 1.0), '132': (0.0784313725490196, 0.0, 1.0, 1.0), '100': (1.0, 0.4392156862745098, 0.0, 1.0), '83': (1.0, 0.6784313725490196, 0.00392156862745098, 1.0), '85': (0.996078431372549, 0.7215686274509804, 0.7215686274509804, 1.0), '79': (1.0, 0.3607843137254902, 0.0, 1.0), '144': (0.0, 0.3607843137254902, 1.0, 1.0), '113': (0.3568627450980392, 1.0, 0.0, 1.0), '133': (1.0, 1.0, 0.0, 1.0), '52': (0.12156862745098039, 1.0, 0.0, 1.0), '28': (0.8627450980392157, 0.8627450980392157, 0.8627450980392157, 1.0), '10': (0.0196078431372549, 0.9803921568627451, 0.027450980392156862, 1.0), '2': (0.7058823529411765, 0.47058823529411764, 0.47058823529411764, 1.0), '103': (0.6392156862745098, 0.996078431372549, 0.0, 1.0), '91': (0.00392156862745098, 0.996078431372549, 0.3254901960784314, 1.0), '121': (1.0, 0.8, 0.0, 1.0), '86': (0.0, 0.12156862745098039, 0.996078431372549, 1.0), '27': (0.03529411764705882, 0.027450980392156862, 0.9058823529411765, 1.0), '114': (0.0, 0.8823529411764706, 1.0, 1.0), '20': (0.796078431372549, 0.27450980392156865, 0.011764705882352941, 1.0), '134': (0.0, 0.6, 1.0, 1.0), '17': (0.5568627450980392, 1.0, 0.5450980392156862, 1.0), '3': (0.023529411764705882, 0.9019607843137255, 0.9019607843137255, 1.0), '65': (0.00392156862745098, 0.996078431372549, 0.4392156862745098, 1.0), '139': (0.6823529411764706, 0.0, 1.0, 1.0), '130': (0.8392156862745098, 1.0, 0.0, 1.0), '136': (0.00392156862745098, 1.0, 0.803921568627451, 1.0), '51': (0.0784313725490196, 1.0, 0.0, 1.0), '15': (0.03137254901960784, 1.0, 0.20392156862745098, 1.0), '55': (0.6, 1.0, 0.0, 1.0), '5': (0.01568627450980392, 0.7843137254901961, 0.023529411764705882, 1.0), '22': (0.23921568627450981, 0.9019607843137255, 0.984313725490196, 1.0), '72': (0.2, 1.0, 0.0, 1.0), '41': (1.0, 0.4823529411764706, 0.0392156862745098, 1.0), '137': (0.1607843137254902, 0.0, 1.0, 1.0), '92': (0.0, 0.0392156862745098, 1.0, 1.0), '21': (0.0, 0.4, 0.7843137254901961, 1.0), '109': (1.0, 0.0, 0.11764705882352941, 1.0), '1': (0.47058823529411764, 0.47058823529411764, 0.47058823529411764, 1.0), '67': (1.0, 0.00392156862745098, 0.011764705882352941, 1.0), '60': (0.12156862745098039, 0.0, 0.996078431372549, 1.0), '44': (1.0, 0.0196078431372549, 0.6039215686274509, 1.0), '69': (1.0, 0.4, 0.0, 1.0), '107': (0.0, 0.996078431372549, 0.36470588235294116, 1.0), '4': (0.3137254901960784, 0.19607843137254902, 0.20392156862745098, 1.0), '90': (0.0, 1.0, 0.7647058823529411, 1.0), '149': (0.396078431372549, 1.0, 0.0, 1.0), '129': (0.0392156862745098, 0.7490196078431373, 0.8313725490196079, 1.0), '75': (0.0, 1.0, 0.6784313725490196, 1.0), '141': (0.2823529411764706, 0.0, 1.0, 1.0), '128': (1.0, 0.9607843137254902, 0.0, 1.0), '57': (1.0, 0.2784313725490196, 0.0, 1.0), '49': (0.5490196078431373, 0.5490196078431373, 0.5490196078431373, 1.0), '108': (0.7215686274509804, 0.0, 0.996078431372549, 1.0), '87': (0.0, 1.0, 0.23921568627450981, 1.0), '43': (1.0, 0.03137254901960784, 0.16470588235294117, 1.0), '71': (0.0, 0.5607843137254902, 1.0, 1.0), '31': (0.03529411764705882, 0.996078431372549, 0.8352941176470589, 1.0), '81': (1.0, 0.0, 0.9568627450980393, 1.0), '36': (0.027450980392156862, 1.0, 1.0, 1.0), '143': (0.0, 1.0, 0.7254901960784313, 1.0), '148': (0.09411764705882353, 0.7607843137254902, 0.7607843137254902, 1.0), '102': (0.3215686274509804, 0.0, 1.0, 1.0), '84': (1.0, 0.0, 0.08235294117647059, 1.0), '47': (0.6274509803921569, 0.5882352941176471, 0.07450980392156863, 1.0), '127': (1.0, 0.0, 0.47843137254901963, 1.0), '16': (0.996078431372549, 0.023529411764705882, 0.3215686274509804, 1.0), '78': (0.0, 0.996078431372549, 0.6078431372549019, 1.0), '25': (1.0, 0.023529411764705882, 0.27450980392156865, 1.0), '138': (0.16862745098039217, 0.996078431372549, 0.011764705882352941, 1.0), '42': (0.00392156862745098, 1.0, 0.07450980392156863, 1.0), '119': (0.2784313725490196, 1.0, 0.00392156862745098, 1.0), '112': (0.996078431372549, 0.0, 0.4392156862745098, 1.0), '89': (0.996078431372549, 0.0, 0.796078431372549, 1.0), '96': (0.0, 0.47843137254901963, 1.0, 1.0), '37': (0.8823529411764706, 1.0, 0.03529411764705882, 1.0), '48': (0.0, 0.6392156862745098, 0.996078431372549, 1.0), '116': (0.27058823529411763, 0.7176470588235294, 0.6196078431372549, 1.0), '40': (1.0, 0.7607843137254902, 0.027450980392156862, 1.0), '123': (0.0, 1.0, 0.9176470588235294, 1.0)}

def annotation_update(self, context):
    self.color = annotation_colors[self.annotation][:3]

class ObjectADE20KData(bpy.types.PropertyGroup):
    bl_label = "ADE20K Segmentation"
    bl_idname = "dream_textures.ObjectADE20KData"

    enabled: bpy.props.BoolProperty(name="Enabled", default=False)
    annotation: bpy.props.EnumProperty(
        name="Class",
        items=annotation_enum_cases,
        update=annotation_update
    )
    # for visualization only
    color: bpy.props.FloatVectorProperty(name="", subtype='COLOR', default=annotation_colors[annotation_enum_cases[0][0]][:3])

def render_ade20k_map(context, collection=None, invert=True):
    e = threading.Event()
    result = None
    def _execute():
        nonlocal result
        width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
        matrix = context.scene.camera.matrix_world.inverted()
        projection_matrix = context.scene.camera.calc_matrix_camera(
            context,
            x=width,
            y=height
        )
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1)
            
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)

                def render_mesh(mesh, transform, color):
                    mesh.transform(transform)
                    mesh.calc_loop_triangles()
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

                    mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                    mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                    
                    draw_annotation(vertices, indices, color)
                if collection is None:
                    for object in context.object_instances:
                        if not hasattr(object.object, 'dream_textures_ade20k') or not object.object.dream_textures_ade20k.enabled:
                            continue
                        try:
                            mesh = object.object.to_mesh()
                            if mesh is not None:
                                render_mesh(mesh, object.matrix_world, annotation_colors[object.object.dream_textures_ade20k.annotation])
                                object.object.to_mesh_clear()
                        except:
                            continue
                else:
                    for object in collection.objects:
                        if not hasattr(object, 'dream_textures_ade20k') or not object.dream_textures_ade20k.enabled:
                            continue
                        try:
                            mesh = object.to_mesh(depsgraph=context)
                            if mesh is not None:
                                render_mesh(mesh, object.matrix_world, annotation_colors[object.dream_textures_ade20k.annotation])
                                object.to_mesh_clear()
                        except:
                            continue
            result = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
            result[:, :, 3] = 1
        gpu.state.depth_test_set('NONE')
        offscreen.free()
        e.set()
    if threading.current_thread() == threading.main_thread():
        _execute()
        return result
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
        return result

def draw_annotation(vertices, indices, color):
    shader = gpu.shader.from_builtin(UNIFORM_COLOR)
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": vertices},
        indices=indices,
    )
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)