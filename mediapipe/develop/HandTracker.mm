#import "HandTracker.h"
#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPLayerRenderer.h"
#include "mediapipe/framework/formats/landmark.pb.h"

static NSString* const kGraphName = @"hand_tracking_mobile_gpu";
static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
static const char* kLandmarksOutputStream = "hand_landmarks";
static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";
static const char* kNumHandsInputSidePacket = "num_hands";

// Max number of hands to detect/process.
static const int kNumHands = 2;


@interface HandTracker() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@interface Landmark()
- (instancetype)initWithX:(float)x y:(float)y z:(float)z;
@end

@implementation HandTracker {}

#pragma mark - Cleanup methods

- (void)dealloc {
    self.mediapipeGraph.delegate = nil;
    [self.mediapipeGraph cancel];
    // Ignore errors since we're cleaning up.
    [self.mediapipeGraph closeAllInputStreamsWithError:nil];
    [self.mediapipeGraph waitUntilDoneWithError:nil];
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  return newGraph;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
        [self.mediapipeGraph addFrameOutputStream:kOutputStream
                                 outputPacketType:MPPPacketTypePixelBuffer];
        self.mediapipeGraph.delegate = self;
        
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumHands))
                                     named:kNumHandsInputSidePacket];
        [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream
                                 outputPacketType:MPPPacketTypeRaw];
        // Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
        self.mediapipeGraph.maxFramesInFlight = 2;
    }
    return self;
}

- (void)startGraph {
    // Start running self.mediapipeGraph.
    NSError* error;
    if (![self.mediapipeGraph startWithError:&error]) {
        NSLog(@"Failed to start graph: %@", error);
    }
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
  didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
            fromStream:(const std::string&)streamName {
      if (streamName == kOutputStream) {
          [_delegate handTracker: self didOutputPixelBuffer: pixelBuffer];
      }
}

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
       didOutputPacket:(const ::mediapipe::Packet&)packet
            fromStream:(const std::string&)streamName {
//    NSLog(@"mediapipe Graph");
    if (streamName == kLandmarksOutputStream) {
        if (packet.IsEmpty()) {
          NSLog(@"[TS:%lld] No hand landmarks", packet.Timestamp().Value());
          return;
        }
        const auto& multiHandLandmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
//        NSLog(@"[TS:%lld] Number of hand instances with landmarks: %lu", packet.Timestamp().Value(),
//              multiHandLandmarks.size());
        NSMutableArray<NSMutableArray<Landmark *> *> *result = [NSMutableArray array];
        for (int handIndex = 0; handIndex < multiHandLandmarks.size(); ++handIndex) {
            
            NSMutableArray<Landmark *> *hand = [NSMutableArray array];
            for (int i = 0; i < multiHandLandmarks[handIndex].landmark_size(); ++i) {
                Landmark *landmark = [[Landmark alloc] initWithX:multiHandLandmarks[handIndex].landmark(i).x()
                                                               y:multiHandLandmarks[handIndex].landmark(i).y()
                                                               z:multiHandLandmarks[handIndex].landmark(i).z()];
                [hand addObject:landmark];
            }
            [result addObject:hand];
        }
        [_delegate handTracker: self didOutputLandmarks: result];
    }
}

- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer {
    [self.mediapipeGraph sendPixelBuffer:imageBuffer
                              intoStream:kInputStream
                              packetType:MPPPacketTypePixelBuffer];
}

@end


@implementation Landmark

- (instancetype)initWithX:(float)x y:(float)y z:(float)z
{
    self = [super init];
    if (self) {
        _x = x;
        _y = y;
        _z = z;
    }
    return self;
}

@end
