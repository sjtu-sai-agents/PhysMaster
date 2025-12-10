const bgCol = "#F2F0E7";
const accentCol = "#0000FF";

hljs.initHighlightingOnLoad();

const updateTargetDims = () => {
  return [windowWidth * 0.6, windowHeight * 1]; 
};

const setNodeInfo = (index) => {

  const codeElm = document.getElementById("code");
  const theoElm = document.getElementById("theo_response");
  const critElm = document.getElementById("crit_response");

  const infoElm = document.getElementById("node-info");

  if (codeElm) {
    codeElm.innerHTML = hljs.highlight(
      treeStructData.code[index],
      { language: "python" }
    ).value;
  }

  if (theoElm) {
    theoElm.textContent = treeStructData.theo_response[index];
  }
 
  if (critElm) {
    critElm.textContent = treeStructData.crit_response[index];
  }

  if (infoElm) {
    infoElm.innerHTML = `
      <b>subtask_id:</b> ${treeStructData.subtask_id[index]}<br>
      <b>node_index:</b> ${treeStructData.node_index[index]}<br>
      <b>node_type:</b> ${treeStructData.node_type[index]}<br>
      <b>status:</b> ${treeStructData.status[index]}<br>
      <b>visits:</b> ${treeStructData.visits[index]}<br>
      <b>total_reward:</b> ${treeStructData.total_reward[index]}<br>
      <b>average_reward:</b> ${treeStructData.average_reward[index]}<br>
    `;
  }
};
 

windowResized = () => {
  resizeCanvas(...updateTargetDims());
  awaitingPostResizeOps = true;
};

let scaleFactor = 0.57;
let manualSelection = false;
let currentElemInd = 0;

let treeStructData = <placeholder>

let nodes = [];
let edges = [];

// 拖动画布的变量
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let startDragX = 0;
let startDragY = 0;

setup = () => {
  canvas = createCanvas(...updateTargetDims());
  canvas.mousePressed(startDrag);
  canvas.mouseReleased(stopDrag);
};

function startDrag() {
  dragging = true;
  startDragX = mouseX - offsetX;
  startDragY = mouseY - offsetY;
}

function stopDrag() {
  dragging = false;
}

function keyPressed() {
  if (key === 'Z' || key === 'z') {
    scaleFactor = min(scaleFactor * 1.1, 2.0); // 放大，最多 2 倍
  } else if (key === 'X' || key === 'x') {
    scaleFactor = max(scaleFactor * 0.9, 0.3); // 缩小，最小 0.3 倍
  }
}

class Node {
  x;
  y;
  size;
  xT;
  yT;
  xB;
  yB;
  treeInd;
  color;
  relSize;
  isStatic = false;
  hasChildren = false;
  isRootNode = true;
  isStarred = false;
  selected = false;
  renderSize = 10;
  edges = [];
  bgCol;

  constructor(x, y, relSize, treeInd) {
    const minSize = 60; // 节点最小尺寸
    const maxSize = 120; // 节点最大尺寸

    const maxColor = 10;
    const minColor = 125;

    this.relSize = relSize;
    this.treeInd = treeInd;
    this.size = minSize + (maxSize - minSize) * relSize;
    this.color = minColor + (maxColor - minColor) * relSize;
    this.bgCol = Math.round(Math.max(this.color / 2, 0));

    this.x = x;
    this.y = y;
    this.xT = x;
    this.yT = y - this.size / 2;
    this.xB = x;
    this.yB = y + this.size / 2;

    nodes.push(this);
  }

  child = (node) => {
    let edge = new Edge(this, node);
    this.edges.push(edge);
    edges.push(edge);
    this.hasChildren = true;
    node.isRootNode = false;
    return node;
  };

  render = () => {
    const mouseXlocalCoords = (mouseX - width / 2 - offsetX) / scaleFactor;
    const mouseYlocalCoords = (mouseY - height / 2 - offsetY) / scaleFactor;
    const isMouseOver =
      dist(mouseXlocalCoords, mouseYlocalCoords, this.x, this.y) <
      this.renderSize / 1.5;
    if (isMouseOver) {
      this.renderSize = this.size * 1.1;
      cursor(HAND);
    }
    // if (isMouseOver) cursor(HAND);
    if (isMouseOver && mouseIsPressed) {
      nodes.forEach((n) => (n.selected = false));
      this.selected = true;
      setNodeInfo(this.treeInd);
      manualSelection = true;
    }

    this.renderSize = this.size;

    if (this.selected) {
      fill("#4F8CFF"); // 霓虹蓝
    } else {
      fill("#FF9F43"); // 高级橙
    }


    noStroke();
    drawingContext.shadowBlur = this.selected ? 20 : 8;
    drawingContext.shadowColor = this.selected ? "#4F8CFF" : "#FF9F43";
    
    square(
      this.x - this.renderSize / 2,
      this.y - this.renderSize / 2,
      this.renderSize,
      10,
    );

    noStroke();
    textAlign(CENTER, CENTER);
    textSize(this.renderSize * 0.8); // 更大文字比例
    fill(255);
    text(`${treeStructData.node_index[this.treeInd]}`, this.x, this.y - 1);
  };
}

class Edge {
  nodeT;
  nodeB;
  weight = 0;

  constructor(nodeT, nodeB) {
    this.nodeT = nodeT;
    this.nodeB = nodeB;
    this.weight = 3; // 固定边粗细
  }

  color = () => this.nodeB.color;

  render = () => {
    strokeWeight(this.weight);
    stroke(0); // 黑色线条
    noFill();
    bezier(
      this.nodeT.xB,
      this.nodeT.yB,
      this.nodeT.xB,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.nodeB.xT,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.nodeB.xT,
      this.nodeB.yT,
    );
  };
}

draw = () => {

  cursor(ARROW);
  frameRate(5);

  if (nodes.length == 0) {
    const spacingHeight = height * 1.3;
    const spacingWidth = width * 1.3;

    treeStructData.layout.forEach((lay, index) => {
      const rel = parseFloat(treeStructData.average_reward[index]) || 0;

      new Node(
        spacingWidth * lay[0] - spacingWidth / 2,
        20 + spacingHeight * lay[1] - spacingHeight / 2,
        1 - rel,
        index,
      );
    });

    treeStructData.edges.forEach((ind) => {
      nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.isStatic = true;
    });
    nodes[0].selected = true;
    setNodeInfo(0);
  }

  const staticNodes = nodes.filter((n) => n.isStatic);
  if (staticNodes.length > 0) {
    const largestNode = staticNodes.reduce((prev, current) =>
      prev.relSize > current.relSize ? prev : current,
    );
    if (!manualSelection) {
      if (!largestNode.selected) {
        setNodeInfo(this.treeInd);
      }
      staticNodes.forEach((node) => {
        node.selected = node === largestNode;
      });
    }
  }
  background(bgCol);

  if (dragging) {
    offsetX = mouseX - startDragX;
    offsetY = mouseY - startDragY;
  }

  translate(width / 2 + offsetX, height / 2 + offsetY);
  scale(scaleFactor);

  edges.forEach((e) => e.render());
  nodes.forEach((n) => n.render());

};
