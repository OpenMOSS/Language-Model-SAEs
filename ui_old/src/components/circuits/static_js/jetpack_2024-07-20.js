// Forked from https://github.com/gka/d3-jetpack — BSD-3


(function(){
  function parseAttributes(name) {
    if (typeof name === "string") {
      var attr = {},
        parts = name.split(/([\.#])/g), p;
        name = parts.shift();
      while ((p = parts.shift())) {
        if (p == '.') attr['class'] = attr['class'] ? attr['class'] + ' ' + parts.shift() : parts.shift();
        else if (p == '#') attr.id = parts.shift();
      }
      return {tag: name, attr: attr};
    }
    return name;
  }

  d3.selection.prototype.selectAppend = function(name) {
    var select = d3.selector(name),
       n = parseAttributes(name), s;

    name = d3.creator(n.tag);

    s = this.select(function() {
      return select.apply(this, arguments) ||
          this.appendChild(name.apply(this, arguments));
    });

    for (var key in n.attr) { s.attr(key, n.attr[key]); }
    return s;
  }

  d3.selection.prototype.appendMany = function(name, data){
    return this.selectAll(name).data(data).join(name)
  }

  d3.selection.prototype.append = function(name) {
    var create, n;

    if (typeof name === "function"){
      create = name;
    } else {
      n = parseAttributes(name)
      create = d3.creator(n.tag);
    }
    var sel = this.select(function(){
      return this.appendChild(create.apply(this, arguments))
    })

    if (n) for (var key in n.attr) { sel.attr(key, n.attr[key]) }
    return sel
  }

  d3.selection.prototype.at = function(name, value) {
    if (typeof(name) == 'object'){
      for (var key in name){
        this.attr(camelCaseAttrs.test(key) ? key.replace(/([a-z\d])([A-Z])/g, '$1-$2').toLowerCase() : key, name[key]);
      }
      return this;
    } else{
      return arguments.length == 1 ? this.attr(name) : this.attr(name, value);
    }
  }
  const camelCaseAttrs = /^(alignmentBaseline|allowReorder|attributeName|attributeType|autoReverse|baseFrequency|baselineShift|baseProfile|calcMode|clipPathUnits|clipRule|colorInterpolation|colorInterpolationFilters|colorProfile|contentScriptType|contentStyleType|diffuseConstant|dominantBaseline|edgeMode|enableBackground|externalResourcesRequired|fillOpacity|fillRule|filterRes|filterUnits|floodColor|floodOpacity|fontFamily|fontSize|fontSizeAdjust|fontStretch|fontStyle|fontVariant|fontWeight|glyphOrientationHorizontal|glyphOrientationVertical|glyphRef|gradientTransform|gradientUnits|imageRendering|kernelMatrix|kernelUnitLength|kerning|keyPoints|keySplines|keyTimes|lengthAdjust|letterSpacing|lightingColor|limitingConeAngle|markerEnd|markerMid|markerStart|maskContentUnits|maskUnits|midMarker|numOctaves|overlinePosition|overlineThickness|paintOrder|pathLength|patternContentUnits|patternTransform|patternUnits|pointerEvents|pointsAtX|pointsAtY|pointsAtZ|preserveAlpha|preserveAspectRatio|primitiveUnits|referrerPolicy|repeatCount|repeatDur|requiredExtensions|requiredFeatures|shapeRendering|specularConstant|specularExponent|spreadMethod|startOffset|stdDeviation|stopColor|stopOpacity|stitchTiles|strikethroughPosition|strikethroughThickness|strokeDasharray|strokeDashoffset|strokeLinecap|strokeLinejoin|strokeMiterlimit|strokeOpacity|strokeWidth|surfaceScale|systemLanguage|tableValues|targetX|targetY|textAnchor|textDecoration|textLength|textRendering|underlinePosition|underlineThickness|vectorEffect|viewTarget|wordSpacing|writingMode|xChannelSelector|yChannelSelector|zoomAndPan)$/

  d3.selection.prototype.st = function(name, value) {
    if (typeof(name) == 'object'){
      for (var key in name){
        addStyle(this, key, name[key]);
      }
      return this;
    } else {
      return arguments.length == 1 ? this.style(name) : addStyle(this, name, value);
    }

    function addStyle(sel, style, value){
      style = style.replace(/([a-z\d])([A-Z])/g, '$1-$2').toLowerCase();

      var pxStyles = 'top left bottom right padding-top padding-left padding-bottom padding-right border-top border-left-width border-bottom-width border-right-width margin-top margin-left margin-bottom margin-right font-size width stroke-width line-height margin padding border border-radius max-width min-width max-height min-height gap';

      if (~pxStyles.indexOf(style) ){
        sel.style(style, typeof value == 'function' ? wrapPx(value) : addPx(value));
      } else{
        sel.style(style, value);
      }

      return sel;
    }

    function addPx(d){ return d.match ? d : d + 'px'; }
    function wrapPx(fn){
      return function(){
        var val = fn.apply(this, arguments)
        return addPx(val)
      }

    }
  }

  d3.selection.prototype.translate = function(xy, dim) {
    var node = this.node()
    return !node ? this : node.getBBox ?
      this.attr('transform', function(d,i) {
        var p = typeof xy == 'function' ? xy.call(this, d,i) : xy;
        if (dim === 0) p = [p, 0]; else if (dim === 1) p = [0, p];
        return 'translate(' + p[0] +','+ p[1]+')';
      }) :
      this.style('transform', function(d,i) {
        var p = typeof xy == 'function' ? xy.call(this, d,i) : xy;
        if (dim === 0) p = [p, 0]; else if (dim === 1) p = [0, p];
        return 'translate(' + p[0] +'px,'+ p[1]+'px)';
      });
  }

  d3.selection.prototype.parent = function() {
    var parents = [];
    return this.filter(function() {
      if (parents.indexOf(this.parentNode) > -1) return false;
      parents.push(this.parentNode);
      return true;
    }).select(function() {
      return this.parentNode;
    });
  }

  d3.nestBy = function(array, key){
    return d3.groups(array, key).map(function([key, values]){
      values.key = key;
      return values;
    });
  }

  d3.clamp = function(min, d, max) {
    return Math.max(min, Math.min(max, d));
  }


  d3.conventions = function(c){
    c = c || {};

    c.margin = c.margin || {}
    ;['top', 'right', 'bottom', 'left'].forEach(function(d){
      if (!c.margin[d] && c.margin[d] !== 0) c.margin[d] = 20 ;
    });

    if (c.parentSel) c.sel = c.parentSel // backwords comp
    var node = c.sel && c.sel.node()

    c.totalWidth  = c.totalWidth  || node && node.offsetWidth  || 960;
    c.totalHeight = c.totalHeight || node && node.offsetHeight || 500;

    c.width  = c.width  || c.totalWidth  - c.margin.left - c.margin.right;
    c.height = c.height || c.totalHeight - c.margin.top - c.margin.bottom;

    c.totalWidth = c.width + c.margin.left + c.margin.right;
    c.totalHeight = c.height + c.margin.top + c.margin.bottom;

    c.sel = c.sel || select('body');
    c.sel.st({position: 'relative', height: c.totalHeight, width: c.totalWidth})

    c.x = c.x || d3.scaleLinear().range([0, c.width]);
    c.y = c.y || d3.scaleLinear().range([c.height, 0]);

    c.xAxis = c.xAxis || d3.axisBottom().scale(c.x);
    c.yAxis = c.yAxis || d3.axisLeft().scale(c.y);

    c.layers = (c.layers || 's').split('').map(function(type){
      var layer
      if (type == 's'){
        layer = c.sel.append('svg')
            .st({position: c.layers ? 'absolute' : ''})
            .attr('width', c.totalWidth)
            .attr('height', c.totalHeight)
          .append('g')
            .attr('transform', 'translate(' + c.margin.left + ',' + c.margin.top + ')');

        if (!c.svg) c.svg = layer // defaults to lowest svg layer
      } else if (type == 'c'){
        var s = window.devicePixelRatio || 1

        layer = c.sel.append('canvas')
          .at({width: c.totalWidth*s, height: c.totalHeight*s})
          .st({width: c.totalWidth, height: c.totalHeight})
          .st({position: 'absolute'})
          .node().getContext('2d')
        layer.scale(s, s)
        layer.translate(c.margin.left, c.margin.top)
      } else if (type == 'd'){
        layer = c.sel.append('div')
          .st({
            position: 'absolute',
            left: c.margin.left,
            top: c.margin.top,
            width: c.width,
            height: c.height
          });
      }

      return layer
    })

    c.drawAxis = svg => {
      if (!svg) svg = c.svg
      var xAxisSel = svg.append('g')
          .attr('class', 'x axis')
          .attr('transform', 'translate(0,' + c.height + ')')
          .call(c.xAxis);

      var yAxisSel = svg.append('g')
          .attr('class', 'y axis')
          .call(c.yAxis);

      return {xAxisSel: xAxisSel, yAxisSel: yAxisSel}
    }

    return c;
  }

  d3.attachTooltip = function(sel, tooltipSel, fieldFns){
    if (!sel.size()) return;

    tooltipSel = tooltipSel || d3.select('.tooltip');

    sel
        .on('mouseover.attachTooltip', ttDisplay)
        .on('mousemove.attachTooltip', ttMove)
        .on('mouseout.attachTooltip',  ttHide)
        .on('click.attachTooltip', (_, d) => console.log(d))

    var d = sel.datum();
    fieldFns = fieldFns || Object.keys(d)
        .filter(function(str){
          return (typeof d[str] != 'object') && (d[str] != 'array');
        })
        .map(function(str){
          return function(d){ return str + ': <b>' + d[str] + '</b>'; };
        });

    function ttDisplay(e, d){
      tooltipSel
          .classed('tooltip-hidden', false)
          .html('')
        .appendMany('div', fieldFns)
          .html(function(fn){ return fn(d); });

      d3.select(this).classed('tooltipped', true);
    }

    function ttMove(e, d){
      if (!tooltipSel.size()) return;

      var x = e.clientX,
          y = e.clientY,
          bb = tooltipSel.node().getBoundingClientRect(),
          left = d3.clamp(20, (x-bb.width/2), window.innerWidth - bb.width - 20),
          top = innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

      tooltipSel
        .style('left', left +'px')
        .style('top', top + 'px');
    }

    function ttHide(e, d){
      tooltipSel.classed('tooltip-hidden', true);

      d3.selectAll('.tooltipped').classed('tooltipped', false);
    }
  }
})()
